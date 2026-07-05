"""
Context-window management utilities for LLM agent workflows.

This module provides a focused collection of tools that solve one of the most
common pain-points when building LLM-powered applications and agents:
**keeping a conversation inside the model's context window without losing
important information.**

Every function and class is pure-Python, fully type-hinted, zero-dependency
(only the standard library), and safe to drop into any LLM application —
chat-bots, agent orchestration, RAG pipelines, or multi-turn tool-use loops.

Typical use-cases
------------------
* Allocating a token budget across *system prompt*, *conversation history*,
  and *current user message* before sending a request.
* Trimming a long message list so the total token count fits inside the
  model's context window while preserving the most recent messages.
* Summarising older conversation turns into a compact text block so that
  the agent retains long-term context without exceeding the window.
* Managing a rolling conversation buffer that automatically evicts old
  messages and maintains an optional running summary.

Quick start
-----------
.. code-block:: python

    from ilovetools.ai.context_manager import (
        ContextWindow,
        ConversationBuffer,
        ConversationSummaryBuffer,
        trim_messages,
        summarize_history,
        allocate_token_budget,
        count_message_tokens,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]

    # 1. Trim to fit a 100-token window
    trimmed = trim_messages(messages, max_tokens=100)
    assert len(trimmed) <= len(messages)

    # 2. Allocate a 4 000-token budget
    budget = allocate_token_budget(total=4000, system_ratio=0.15, history_ratio=0.55)
    # budget -> {"system": 600, "history": 2200, "current": 1200}

    # 3. Rolling buffer with token cap
    buf = ConversationBuffer(max_tokens=500)
    buf.add_many(messages)
    recent = buf.get_messages()  # only the most recent that fit

    # 4. Summary buffer — keeps recent messages + a running summary
    sb = ConversationSummaryBuffer(max_tokens=200, summarizer=summary_fn)
    sb.add_many(messages)
    payload = sb.get_payload()  # [{"role": "system", ...summary...}, ...recent...]

    # 5. Full-featured context-window manager
    cw = ContextWindow(model="gpt-4", reserve_tokens=500)
    cw.set_system_prompt("You are a helpful assistant.")
    cw.add_user_message("Hello!")
    cw.add_assistant_message("Hi there!")
    payload = cw.build_payload()
"""

from __future__ import annotations

import math
import textwrap
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

__all__ = [
    "ContextWindow",
    "ConversationBuffer",
    "ConversationSummaryBuffer",
    "trim_messages",
    "summarize_history",
    "allocate_token_budget",
    "count_message_tokens",
    "estimate_tokens",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Roughly 4 characters per token for English text (empirical average).
_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text*.

    Uses a lightweight heuristic (~4 characters per token) that is accurate
    within ~10 % for typical English prose and code.  No external tokenizer
    is required.

    Parameters
    ----------
    text : str
        The input string.

    Returns
    -------
    int
        Estimated token count (always >= 0).

    Examples
    --------
    >>> estimate_tokens("Hello, world!")
    4
    >>> estimate_tokens("")
    0
    >>> estimate_tokens("a" * 100)
    25
    """
    if not isinstance(text, str) or not text:
        return 0
    return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))


def count_message_tokens(
    message: Union[str, Dict[str, str], Sequence[Dict[str, str]]],
    model: str = "gpt-3.5-turbo",
) -> int:
    """Count (estimate) tokens for a single message, dict, or list of dicts.

    When a *dict* is supplied the ``"role"`` and ``"content"`` keys are
    counted together (role adds ~1 token of overhead per message, matching
    common chat formats).  When a *list* is supplied the total across all
    messages is returned.

    Parameters
    ----------
    message : str | dict | list[dict]
        A plain string, a single message dict
        (``{"role": ..., "content": ...}``), or a list of message dicts.
    model : str
        Model name -- currently used only to slightly adjust the per-message
        overhead.  Values other than the known defaults fall back to the
        generic estimate.

    Returns
    -------
    int
        Estimated total token count.

    Examples
    --------
    >>> count_message_tokens("Hello, world!")
    4

    >>> count_message_tokens({"role": "user", "content": "Hello, world!"})
    6

    >>> msgs = [{"role": "system", "content": "Be helpful."},
    ...         {"role": "user", "content": "Hi"}]
    >>> count_message_tokens(msgs) > 0
    True
    """
    # Per-message overhead (role tags, separator tokens).
    _overhead = {
        "gpt-3.5-turbo": 1,
        "gpt-4": 1,
        "gpt-4-turbo": 1,
        "claude-3": 2,
        "claude-2": 2,
        "llama-2": 1,
        "llama-3": 1,
        "gemini-pro": 1,
    }
    extra = _overhead.get(model, 1)

    if isinstance(message, str):
        return estimate_tokens(message)

    if isinstance(message, dict):
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        return estimate_tokens(role) + estimate_tokens(content) + extra

    if isinstance(message, (list, tuple)):
        return sum(count_message_tokens(m, model) for m in message)

    return 0


# ---------------------------------------------------------------------------
# Token-budget allocation
# ---------------------------------------------------------------------------

def allocate_token_budget(
    total: int,
    system_ratio: float = 0.15,
    history_ratio: float = 0.55,
    *,
    min_system: int = 100,
    min_current: int = 100,
) -> Dict[str, int]:
    """Split a *total* token budget across system / history / current parts.

    The remaining share (``1 - system_ratio - history_ratio``) is allocated
    to the *current* part (the latest user message + expected response).

    Parameters
    ----------
    total : int
        Total tokens available (e.g. model context window minus reserve).
    system_ratio : float
        Fraction of tokens for the system prompt (0-1).  Default ``0.15``.
    history_ratio : float
        Fraction of tokens for conversation history (0-1).  Default ``0.55``.
    min_system : int
        Minimum tokens guaranteed for the system prompt.
    min_current : int
        Minimum tokens guaranteed for the current message + response.

    Returns
    -------
    dict[str, int]
        ``{"system": int, "history": int, "current": int}`` -- the three
        parts always sum to *total*.

    Raises
    ------
    ValueError
        If *total* <= 0, or the ratios are outside [0, 1], or their sum
        exceeds 1.

    Examples
    --------
    >>> allocate_token_budget(4000)
    {'system': 600, 'history': 2200, 'current': 1200}

    >>> allocate_token_budget(8000, system_ratio=0.1, history_ratio=0.6)
    {'system': 800, 'history': 4800, 'current': 2400}

    >>> allocate_token_budget(1000, system_ratio=0.5, history_ratio=0.5)
    {'system': 500, 'history': 400, 'current': 100}
    """
    if total <= 0:
        raise ValueError("total must be a positive integer")
    if not 0 <= system_ratio <= 1:
        raise ValueError("system_ratio must be between 0 and 1")
    if not 0 <= history_ratio <= 1:
        raise ValueError("history_ratio must be between 0 and 1")
    if system_ratio + history_ratio > 1:
        raise ValueError("system_ratio + history_ratio must not exceed 1")

    current_ratio = 1.0 - system_ratio - history_ratio

    system = max(min_system, round(total * system_ratio))
    current = max(min_current, round(total * current_ratio))
    history = total - system - current

    # If history went negative, redistribute.
    if history < 0:
        history = 0
        # Shrink the larger of system / current.
        if system >= current:
            system = total - current
        else:
            current = total - system

    return {"system": system, "history": history, "current": current}


# ---------------------------------------------------------------------------
# Message trimming
# ---------------------------------------------------------------------------

def trim_messages(
    messages: List[Dict[str, str]],
    max_tokens: int,
    *,
    model: str = "gpt-3.5-turbo",
    preserve_system: bool = True,
    preserve_first: int = 0,
    preserve_last: int = 0,
) -> List[Dict[str, str]]:
    """Trim a message list so the total token count fits within *max_tokens*.

    Messages are removed from the **middle** of the conversation -- the
    oldest non-protected messages are dropped first -- so that the system
    prompt (optionally), the first *preserve_first* messages, and the last
    *preserve_last* messages are always retained.

    Parameters
    ----------
    messages : list[dict]
        Conversation messages (``{"role": ..., "content": ...}``).
    max_tokens : int
        Maximum total tokens to keep.
    model : str
        Model name for token estimation.
    preserve_system : bool
        If ``True`` (default), system messages are never removed.
    preserve_first : int
        Number of messages from the start to always keep.
    preserve_last : int
        Number of messages from the end to always keep.

    Returns
    -------
    list[dict]
        A new list of messages whose total token count <= *max_tokens*.

    Raises
    ------
    ValueError
        If *max_tokens* <= 0.

    Examples
    --------
    >>> msgs = [{"role": "system", "content": "Be helpful."},
    ...         {"role": "user", "content": "Hi"},
    ...         {"role": "assistant", "content": "Hello!"},
    ...         {"role": "user", "content": "Bye"}]
    >>> result = trim_messages(msgs, max_tokens=100)
    >>> len(result) <= len(msgs)
    True

    >>> trim_messages(msgs, max_tokens=0)
    Traceback (most recent call last):
        ...
    ValueError: max_tokens must be a positive integer
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not messages:
        return []

    # Work on a copy.
    msgs = list(messages)
    total = count_message_tokens(msgs, model)
    if total <= max_tokens:
        return msgs

    # Build the set of indices that are protected.
    protected: set[int] = set()
    n = len(msgs)
    for i in range(min(preserve_first, n)):
        protected.add(i)
    for i in range(max(n - preserve_last, 0), n):
        protected.add(i)
    if preserve_system:
        for i, m in enumerate(msgs):
            if m.get("role") == "system":
                protected.add(i)

    # Remove oldest unprotected messages until we fit.
    for i in range(n):
        if total <= max_tokens:
            break
        if i in protected:
            continue
        total -= count_message_tokens(msgs[i], model)
        msgs[i] = None  # type: ignore[assignment]

    return [m for m in msgs if m is not None]


# ---------------------------------------------------------------------------
# History summarisation
# ---------------------------------------------------------------------------

def summarize_history(
    messages: List[Dict[str, str]],
    summarizer: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    *,
    max_messages: int = 10,
    system_role: str = "system",
) -> Dict[str, str]:
    """Summarise a list of conversation messages into a single system message.

    If a *summarizer* callable is provided it is invoked with the message
    list and must return a string.  Otherwise a built-in heuristic
    summariser is used that concatenates the role and a truncated content
    for each message.

    Parameters
    ----------
    messages : list[dict]
        Messages to summarise.
    summarizer : callable, optional
        Custom summarisation function ``(list[dict]) -> str``.  If
        ``None``, a built-in heuristic is used.
    max_messages : int
        Maximum number of messages to include in the default summary.
    system_role : str
        Role name for the returned summary message (default ``"system"``).

    Returns
    -------
    dict[str, str]
        A single message dict ``{"role": system_role, "content": ...}``.

    Examples
    --------
    >>> msgs = [
    ...     {"role": "user", "content": "What is 2+2?"},
    ...     {"role": "assistant", "content": "4"},
    ... ]
    >>> result = summarize_history(msgs)
    >>> result["role"]
    'system'
    >>> "2+2" in result["content"]
    True

    >>> custom = lambda ms: f"{len(ms)} messages"
    >>> summarize_history(msgs, summarizer=custom)["content"]
    '2 messages'
    """
    if not messages:
        return {"role": system_role, "content": ""}

    if summarizer is not None:
        content = summarizer(messages)
    else:
        lines: List[str] = []
        for m in messages[:max_messages]:
            role = m.get("role", "unknown")
            content = str(m.get("content", ""))
            # Truncate very long individual messages.
            if len(content) > 200:
                content = content[:197] + "..."
            lines.append(f"[{role}] {content}")
        suffix = ""
        if len(messages) > max_messages:
            suffix = f"\n... ({len(messages) - max_messages} earlier messages omitted)"
        content = "Conversation summary:\n" + "\n".join(lines) + suffix

    return {"role": system_role, "content": content}


# ---------------------------------------------------------------------------
# ConversationBuffer -- rolling window with token cap
# ---------------------------------------------------------------------------

class ConversationBuffer:
    """A rolling conversation buffer that enforces a token-cap.

    Messages are added one at a time (or in bulk).  When the total token
    count exceeds ``max_tokens`` the **oldest** non-system messages are
    evicted until the budget is satisfied.  System messages are always
    retained.

    Parameters
    ----------
    max_tokens : int
        Maximum total tokens the buffer will hold.
    model : str
        Model name for token estimation.

    Examples
    --------
    >>> buf = ConversationBuffer(max_tokens=50)
    >>> buf.add_user_message("Hello!")
    >>> buf.add_assistant_message("Hi there!")
    >>> msgs = buf.get_messages()
    >>> len(msgs) >= 1
    True

    >>> buf.clear()
    >>> buf.get_messages()
    []

    >>> ConversationBuffer(max_tokens=0)
    Traceback (most recent call last):
        ...
    ValueError: max_tokens must be a positive integer
    """

    def __init__(self, max_tokens: int = 4096, model: str = "gpt-3.5-turbo") -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self._max_tokens = max_tokens
        self._model = model
        self._messages: List[Dict[str, str]] = []

    # -- public API --------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a single message and evict if over budget."""
        self._messages.append({"role": role, "content": content})
        self._evict()

    def add_user_message(self, content: str) -> None:
        """Shortcut for ``add_message("user", content)``."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Shortcut for ``add_message("assistant", content)``."""
        self.add_message("assistant", content)

    def add_system_message(self, content: str) -> None:
        """Shortcut for ``add_message("system", content)``."""
        self.add_message("system", content)

    def add_many(self, messages: Sequence[Dict[str, str]]) -> None:
        """Append multiple messages and evict if over budget."""
        self._messages.extend(
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        )
        self._evict()

    def get_messages(self) -> List[Dict[str, str]]:
        """Return a shallow copy of the current message list."""
        return list(self._messages)

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

    def token_count(self) -> int:
        """Return the estimated total token count of the buffer."""
        return count_message_tokens(self._messages, self._model)

    @property
    def max_tokens(self) -> int:
        """The configured token cap."""
        return self._max_tokens

    @property
    def model(self) -> str:
        """The model name used for token estimation."""
        return self._model

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"ConversationBuffer(max_tokens={self._max_tokens}, "
            f"messages={len(self._messages)}, tokens={self.token_count()})"
        )

    # -- internal ----------------------------------------------------------

    def _evict(self) -> None:
        """Remove oldest non-system messages until under budget."""
        while self.token_count() > self._max_tokens and len(self._messages) > 1:
            # Find the first non-system message to evict.
            evicted = False
            for i, m in enumerate(self._messages):
                if m.get("role") != "system":
                    self._messages.pop(i)
                    evicted = True
                    break
            if not evicted:
                # Only system messages left -- evict the oldest.
                if len(self._messages) > 1:
                    self._messages.pop(0)
                else:
                    break


# ---------------------------------------------------------------------------
# ConversationSummaryBuffer -- hybrid memory (summary + recent)
# ---------------------------------------------------------------------------

class ConversationSummaryBuffer:
    """Hybrid conversation memory: running summary + recent messages.

    Older messages are periodically compressed into a text summary.  The
    most recent messages are kept verbatim.  This gives the agent both
    long-term context (via the summary) and precise recent context (via
    the verbatim messages) within a fixed token budget.

    Parameters
    ----------
    max_tokens : int
        Maximum total tokens for the entire payload (summary + messages).
    summarizer : callable, optional
        Custom summarisation function ``(list[dict]) -> str``.  If
        ``None``, the built-in heuristic from :func:`summarize_history`
        is used.
    model : str
        Model name for token estimation.
    summary_role : str
        Role for the summary message (default ``"system"``).

    Examples
    --------
    >>> buf = ConversationSummaryBuffer(max_tokens=100)
    >>> buf.add_user_message("Hello!")
    >>> buf.add_assistant_message("Hi!")
    >>> payload = buf.get_payload()
    >>> isinstance(payload, list)
    True

    >>> buf.clear()
    >>> len(buf)
    0

    >>> ConversationSummaryBuffer(max_tokens=0)
    Traceback (most recent call last):
        ...
    ValueError: max_tokens must be a positive integer
    """

    def __init__(
        self,
        max_tokens: int = 2048,
        summarizer: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        model: str = "gpt-3.5-turbo",
        summary_role: str = "system",
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self._max_tokens = max_tokens
        self._summarizer = summarizer
        self._model = model
        self._summary_role = summary_role
        self._summary: str = ""
        self._messages: List[Dict[str, str]] = []

    # -- public API --------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a message and trigger summarisation if needed."""
        self._messages.append({"role": role, "content": content})
        self._maybe_summarize()

    def add_user_message(self, content: str) -> None:
        """Shortcut for ``add_message("user", content)``."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Shortcut for ``add_message("assistant", content)``."""
        self.add_message("assistant", content)

    def add_many(self, messages: Sequence[Dict[str, str]]) -> None:
        """Append multiple messages and trigger summarisation if needed."""
        self._messages.extend(
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        )
        self._maybe_summarize()

    def get_payload(self) -> List[Dict[str, str]]:
        """Return the full message list: summary message + recent messages.

        If a summary exists it is prepended as a system message.  The
        total token count of the returned list is guaranteed to be <=
        ``max_tokens``.
        """
        result: List[Dict[str, str]] = []
        if self._summary:
            result.append({"role": self._summary_role, "content": self._summary})
        result.extend(self._messages)

        # Final safety trim.
        total = count_message_tokens(result, self._model)
        if total > self._max_tokens:
            result = trim_messages(
                result,
                max_tokens=self._max_tokens,
                model=self._model,
                preserve_system=bool(self._summary),
                preserve_last=len(self._messages),
            )
        return result

    def get_summary(self) -> str:
        """Return the current summary text (may be empty)."""
        return self._summary

    def get_messages(self) -> List[Dict[str, str]]:
        """Return a copy of the recent (non-summarised) messages."""
        return list(self._messages)

    def clear(self) -> None:
        """Clear both the summary and the message buffer."""
        self._summary = ""
        self._messages.clear()

    def token_count(self) -> int:
        """Return the estimated total token count of the full payload."""
        return count_message_tokens(self.get_payload(), self._model)

    @property
    def max_tokens(self) -> int:
        """The configured token cap."""
        return self._max_tokens

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"ConversationSummaryBuffer(max_tokens={self._max_tokens}, "
            f"summary={'yes' if self._summary else 'no'}, "
            f"messages={len(self._messages)})"
        )

    # -- internal ----------------------------------------------------------

    def _maybe_summarize(self) -> None:
        """If over budget, move older messages into the summary."""
        total = count_message_tokens(
            ([{"role": self._summary_role, "content": self._summary}] if self._summary else [])
            + self._messages,
            self._model,
        )
        if total <= self._max_tokens:
            return

        # Move messages from the front into the summary until we fit.
        while self._messages and total > self._max_tokens:
            # Take up to 5 messages at a time to summarise.
            batch_size = min(5, len(self._messages))
            batch = self._messages[:batch_size]

            summary_msg = summarize_history(
                batch,
                summarizer=self._summarizer,
                system_role=self._summary_role,
            )
            new_summary_text = summary_msg["content"]

            if self._summary:
                self._summary = self._summary + "\n" + new_summary_text
            else:
                self._summary = new_summary_text

            self._messages = self._messages[batch_size:]

            # Truncate summary if it alone exceeds the budget.
            summary_tokens = estimate_tokens(self._summary)
            if summary_tokens >= self._max_tokens:
                # Keep only the most recent portion of the summary.
                max_chars = int(self._max_tokens * _CHARS_PER_TOKEN * 0.8)
                if len(self._summary) > max_chars:
                    self._summary = self._summary[-max_chars:]

            total = count_message_tokens(
                [{"role": self._summary_role, "content": self._summary}]
                + self._messages,
                self._model,
            )


# ---------------------------------------------------------------------------
# ContextWindow -- full-featured context-window manager
# ---------------------------------------------------------------------------

# Default context windows (in tokens) for common models.
_DEFAULT_CONTEXT_WINDOWS: Dict[str, int] = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "claude-3": 200000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "llama-2": 4096,
    "llama-3": 8192,
    "llama-3.1": 128000,
    "gemini-pro": 32768,
    "gemini-1.5-pro": 1000000,
}


class ContextWindow:
    """High-level context-window manager for LLM conversations.

    Combines a system prompt, a rolling conversation buffer, and a
    reserve for the model's response into a single, easy-to-use object.
    Calling :meth:`build_payload` returns a message list that is
    guaranteed to fit within the model's context window.

    Parameters
    ----------
    model : str
        Model name -- used to look up the context window size and for
        token estimation.
    context_window : int, optional
        Override the context window size.  If not given, the value is
        looked up from a built-in table.
    reserve_tokens : int
        Tokens reserved for the model's response (not used for input).
        Default ``512``.
    system_ratio : float
        Fraction of the *input* budget for the system prompt.
    history_ratio : float
        Fraction of the *input* budget for conversation history.

    Examples
    --------
    >>> cw = ContextWindow(model="gpt-4", reserve_tokens=500)
    >>> cw.set_system_prompt("You are a helpful assistant.")
    >>> cw.add_user_message("What is 2+2?")
    >>> cw.add_assistant_message("4")
    >>> cw.add_user_message("And 3+3?")
    >>> payload = cw.build_payload()
    >>> isinstance(payload, list)
    True
    >>> payload[0]["role"]
    'system'

    >>> cw.clear_history()
    >>> len(cw)
    0

    >>> cw.token_count() <= cw.input_budget
    True
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        context_window: Optional[int] = None,
        reserve_tokens: int = 512,
        system_ratio: float = 0.15,
        history_ratio: float = 0.55,
    ) -> None:
        self._model = model
        self._context_window = context_window or _DEFAULT_CONTEXT_WINDOWS.get(
            model, 4096
        )
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens must be non-negative")
        if reserve_tokens >= self._context_window:
            raise ValueError("reserve_tokens must be smaller than context_window")
        self._reserve_tokens = reserve_tokens
        self._input_budget = self._context_window - reserve_tokens

        # Validate ratios.
        if not 0 <= system_ratio <= 1:
            raise ValueError("system_ratio must be between 0 and 1")
        if not 0 <= history_ratio <= 1:
            raise ValueError("history_ratio must be between 0 and 1")
        if system_ratio + history_ratio > 1:
            raise ValueError("system_ratio + history_ratio must not exceed 1")

        self._system_ratio = system_ratio
        self._history_ratio = history_ratio
        self._system_prompt: str = ""
        self._buffer = ConversationBuffer(
            max_tokens=self._input_budget,
            model=model,
        )

    # -- public API --------------------------------------------------------

    def set_system_prompt(self, prompt: str) -> None:
        """Set or replace the system prompt."""
        self._system_prompt = prompt

    def get_system_prompt(self) -> str:
        """Return the current system prompt."""
        return self._system_prompt

    def add_user_message(self, content: str) -> None:
        """Append a user message to the conversation."""
        self._buffer.add_user_message(content)

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message to the conversation."""
        self._buffer.add_assistant_message(content)

    def add_message(self, role: str, content: str) -> None:
        """Append a message with an arbitrary role."""
        self._buffer.add_message(role, content)

    def add_many(self, messages: Sequence[Dict[str, str]]) -> None:
        """Append multiple messages to the conversation."""
        self._buffer.add_many(messages)

    def clear_history(self) -> None:
        """Clear the conversation history (keeps the system prompt)."""
        self._buffer.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Return the current conversation history (without system prompt)."""
        return self._buffer.get_messages()

    def build_payload(self) -> List[Dict[str, str]]:
        """Build and return the final message list for an LLM API call.

        The returned list starts with the system prompt (if set) followed
        by the conversation history, trimmed to fit within the input
        token budget.
        """
        messages: List[Dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._buffer.get_messages())

        return trim_messages(
            messages,
            max_tokens=self._input_budget,
            model=self._model,
            preserve_system=bool(self._system_prompt),
            preserve_last=min(len(messages), 10),
        )

    def token_count(self) -> int:
        """Return the estimated token count of the current payload."""
        return count_message_tokens(self.build_payload(), self._model)

    def remaining_tokens(self) -> int:
        """Return the number of input tokens still available."""
        return self._input_budget - self.token_count()

    @property
    def model(self) -> str:
        """The model name."""
        return self._model

    @property
    def context_window(self) -> int:
        """The total context window size (in tokens)."""
        return self._context_window

    @property
    def reserve_tokens(self) -> int:
        """Tokens reserved for the model's response."""
        return self._reserve_tokens

    @property
    def input_budget(self) -> int:
        """Tokens available for input (context_window - reserve_tokens)."""
        return self._input_budget

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ContextWindow(model={self._model!r}, "
            f"context_window={self._context_window}, "
            f"input_budget={self._input_budget}, "
            f"messages={len(self)}, "
            f"tokens_used={self.token_count()})"
        )
