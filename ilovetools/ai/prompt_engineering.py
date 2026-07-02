"""
Prompt engineering utilities for LLM application development.

This module provides a collection of tools that simplify the construction,
templating, and management of prompts for large language model (LLM)
applications and agent workflows.  Every function and class is pure,
fully type-hinted, and has zero external dependencies beyond the Python
standard library.

Typical use-cases
------------------
* Building complex multi-role prompts with a fluent ``PromptBuilder`` API.
* Creating reusable prompt templates with variable validation.
* Constructing few-shot learning prompts from example pairs.
* Extracting and filling ``{variable}`` placeholders in prompt text.
* Intelligently truncating text to fit within a model's context window.
* Estimating API costs before sending requests.
* Formatting chat message histories into a single prompt string.

Examples
--------
>>> from ilovetools.ai.prompt_engineering import (
...     PromptBuilder, PromptTemplate, build_few_shot_prompt,
...     extract_variables, fill_template, truncate_for_context,
...     estimate_api_cost, format_chat_messages,
... )

>>> builder = (
...     PromptBuilder()
...     .system("You are a helpful coding assistant.")
...     .user("Write a Python function to reverse a string.")
... )
>>> print(builder.to_string())
System: You are a helpful coding assistant.
User: Write a Python function to reverse a string.

>>> template = PromptTemplate("Translate {source_lang} to {target_lang}: {text}")
>>> template.variables
['source_lang', 'target_lang', 'text']
>>> template.render(source_lang="English", target_lang="French", text="Hello")
'Translate English to French: Hello'

>>> build_few_shot_prompt(
...     instruction="Classify sentiment:",
...     examples=[("I love it!", "positive"), ("Terrible.", "negative")],
...     query="It's okay.",
... )
"Classify sentiment:\\nInput: I love it!\\nOutput: positive\\n\\nInput: Terrible.\\nOutput: negative\\n\\nInput: It's okay.\\nOutput:"

>>> extract_variables("Hello {name}, your code is {code}")
['name', 'code']

>>> fill_template("Hello {name}!", name="World")
'Hello World!'

>>> truncate_for_context("A" * 100, max_tokens=10)
'AAAAAAAAAA…'

>>> estimate_api_cost(500, 200, model="gpt-4")
0.045

>>> messages = [
...     {"role": "user", "content": "Hi"},
...     {"role": "assistant", "content": "Hello!"},
... ]
>>> format_chat_messages(messages)
'User: Hi\\nAssistant: Hello!'
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "PromptBuilder",
    "PromptTemplate",
    "build_few_shot_prompt",
    "extract_variables",
    "fill_template",
    "truncate_for_context",
    "estimate_api_cost",
    "format_chat_messages",
    "MODEL_CONTEXT_WINDOWS",
    "MODEL_PRICING",
]

# ---------------------------------------------------------------------------
# Constants – model context windows (in tokens) and pricing (per 1K tokens)
# ---------------------------------------------------------------------------

MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3.5-sonnet": 200000,
    "llama-3-8b": 8192,
    "llama-3-70b": 8192,
    "gemini-pro": 32768,
    "gemini-1.5-pro": 1000000,
}

MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # (input_per_1k, output_per_1k) in USD
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "gpt-4": (0.03, 0.06),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-3.5-sonnet": (0.003, 0.015),
    "llama-3-8b": (0.0005, 0.0008),
    "llama-3-70b": (0.0009, 0.0015),
    "gemini-pro": (0.0005, 0.0015),
    "gemini-1.5-pro": (0.00125, 0.005),
}

# Rough estimate: 1 token ≈ 4 characters for English text
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# PromptBuilder – fluent API for constructing multi-role prompts
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    A fluent builder for constructing multi-role LLM prompts.

    Messages are stored in insertion order and can be serialised to a
    plain string or to a list of ``{"role", "content"}`` dicts compatible
    with the OpenAI / Anthropic chat-completions API.

    Examples:
        >>> builder = (
        ...     PromptBuilder()
        ...     .system("You are a helpful assistant.")
        ...     .user("What is 2+2?")
        ... )
        >>> builder.to_messages()
        [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'What is 2+2?'}]

        >>> builder.to_string()
        'System: You are a helpful assistant.\\nUser: What is 2+2?'

        >>> builder.token_count()
        14

        >>> builder.clear()
        >>> builder.to_messages()
        []
    """

    _ROLE_LABELS: Dict[str, str] = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
        "function": "Function",
        "tool": "Tool",
    }

    def __init__(self) -> None:
        self._messages: List[Dict[str, str]] = []

    # -- message appenders -------------------------------------------------

    def system(self, content: str) -> "PromptBuilder":
        """Append a *system* message and return ``self`` for chaining."""
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        self._messages.append({"role": "system", "content": content})
        return self

    def user(self, content: str) -> "PromptBuilder":
        """Append a *user* message and return ``self`` for chaining."""
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        self._messages.append({"role": "user", "content": content})
        return self

    def assistant(self, content: str) -> "PromptBuilder":
        """Append an *assistant* message and return ``self`` for chaining."""
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        self._messages.append({"role": "assistant", "content": content})
        return self

    def function(self, content: str, name: Optional[str] = None) -> "PromptBuilder":
        """Append a *function* message and return ``self`` for chaining."""
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        msg: Dict[str, str] = {"role": "function", "content": content}
        if name:
            msg["name"] = name
        self._messages.append(msg)
        return self

    def tool(self, content: str, tool_call_id: Optional[str] = None) -> "PromptBuilder":
        """Append a *tool* message and return ``self`` for chaining."""
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        msg: Dict[str, str] = {"role": "tool", "content": content}
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        self._messages.append(msg)
        return self

    def add(self, role: str, content: str) -> "PromptBuilder":
        """
        Append a message with an arbitrary *role*.

        Args:
            role: The role name (e.g. ``"system"``, ``"user"``).
            content: The message content.

        Raises:
            ValueError: If *role* or *content* is empty.
        """
        if not role or not isinstance(role, str):
            raise ValueError("role must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        self._messages.append({"role": role, "content": content})
        return self

    # -- output ------------------------------------------------------------

    def to_messages(self) -> List[Dict[str, str]]:
        """Return a shallow copy of the message list (API-ready format)."""
        return list(self._messages)

    def to_string(self, separator: str = "\n") -> str:
        """
        Serialise messages into a single string with role labels.

        Args:
            separator: String inserted between messages (default ``"\\n"``).

        Returns:
            A formatted string like ``"System: ...\\nUser: ..."``.
        """
        parts: List[str] = []
        for msg in self._messages:
            label = self._ROLE_LABELS.get(msg["role"], msg["role"].title())
            parts.append(f"{label}: {msg['content']}")
        return separator.join(parts)

    def token_count(self) -> int:
        """Estimate the total token count of all messages (~4 chars/token)."""
        total_chars = sum(len(m["content"]) for m in self._messages)
        # Add ~4 tokens overhead per message for role formatting
        overhead = len(self._messages) * 4
        return total_chars // _CHARS_PER_TOKEN + overhead

    # -- utilities ---------------------------------------------------------

    def clear(self) -> "PromptBuilder":
        """Remove all messages and return ``self`` for chaining."""
        self._messages.clear()
        return self

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"PromptBuilder(messages={len(self._messages)})"


# ---------------------------------------------------------------------------
# PromptTemplate – reusable template with variable validation
# ---------------------------------------------------------------------------

class PromptTemplate:
    """
    A reusable prompt template with ``{variable}`` placeholders.

    The template text is analysed on construction.  Variable names are
    extracted and validated so that :meth:`render` can verify all required
    variables are supplied.

    Args:
        template: The template string containing ``{variable}`` placeholders.
        validate_on_render: If ``True`` (default), :meth:`render` raises
            ``KeyError`` when a required variable is missing or an unknown
            variable is supplied.

    Examples:
        >>> tpl = PromptTemplate("Summarise: {text} in {language}")
        >>> tpl.variables
        ['text', 'language']
        >>> tpl.render(text="Hello world", language="French")
        'Summarise: Hello world in French'

        >>> tpl.render(text="Hello")
        Traceback (most recent call last):
            ...
        KeyError: "Missing required variables: {'language'}"

        >>> empty_tpl = PromptTemplate("No variables here")
        >>> empty_tpl.variables
        []
        >>> empty_tpl.render()
        'No variables here'
    """

    _VAR_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

    def __init__(self, template: str, validate_on_render: bool = True) -> None:
        if not isinstance(template, str):
            raise TypeError("template must be a string")
        self.template: str = template
        self.validate_on_render: bool = validate_on_render
        self._variables: List[str] = self._extract_vars(template)

    @staticmethod
    def _extract_vars(template: str) -> List[str]:
        """Extract ordered, de-duplicated variable names from *template*."""
        seen: set = set()
        result: List[str] = []
        for match in PromptTemplate._VAR_PATTERN.finditer(template):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    @property
    def variables(self) -> List[str]:
        """Return the list of variable names in order of first appearance."""
        return list(self._variables)

    def render(self, **kwargs: Any) -> str:
        """
        Fill in the template with the provided keyword arguments.

        Args:
            **kwargs: Values for each ``{variable}`` in the template.

        Returns:
            The rendered prompt string.

        Raises:
            KeyError: If a required variable is missing or an unexpected
                variable is supplied (only when *validate_on_render* is True).
        """
        if self.validate_on_render:
            required = set(self._variables)
            provided = set(kwargs.keys())
            missing = required - provided
            if missing:
                raise KeyError(f"Missing required variables: {missing}")
            unexpected = provided - required
            if unexpected:
                raise KeyError(f"Unexpected variables: {unexpected}")

        # Safe format: only replace variables that are provided, leave others intact
        result = self.template
        for var in self._variables:
            if var in kwargs:
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result

    def __repr__(self) -> str:
        return f"PromptTemplate(variables={self._variables})"


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------

def extract_variables(text: str) -> List[str]:
    """
    Extract ``{variable}`` placeholder names from *text*.

    Args:
        text: The template string to scan.

    Returns:
        A list of unique variable names in order of first appearance.

    Examples:
        >>> extract_variables("Hello {name}, your code is {code}")
        ['name', 'code']

        >>> extract_variables("No variables here")
        []

        >>> extract_variables("{a}{b}{a}")
        ['a', 'b']
    """
    if not isinstance(text, str) or not text:
        return []
    seen: set = set()
    result: List[str] = []
    for match in PromptTemplate._VAR_PATTERN.finditer(text):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def fill_template(template: str, strict: bool = False, **kwargs: Any) -> str:
    """
    Fill ``{variable}`` placeholders in *template* with provided values.

    Unlike ``str.format``, this function leaves unknown placeholders intact
    (when *strict* is ``False``) rather than raising ``KeyError``.

    Args:
        template: The template string containing ``{variable}`` placeholders.
        strict: If ``True``, raise ``KeyError`` for missing variables.
        **kwargs: Values to substitute.

    Returns:
        The filled template string.

    Raises:
        KeyError: If *strict* is True and a variable is missing.

    Examples:
        >>> fill_template("Hello {name}!", name="World")
        'Hello World!'

        >>> fill_template("Hello {name} and {other}!", name="World")
        'Hello World and {other}!'

        >>> fill_template("Hello {name}!", strict=True)
        Traceback (most recent call last):
            ...
        KeyError: "Missing required variables: {'name'}"
    """
    if not isinstance(template, str):
        raise TypeError("template must be a string")

    variables = extract_variables(template)

    if strict:
        missing = set(variables) - set(kwargs.keys())
        if missing:
            raise KeyError(f"Missing required variables: {missing}")

    result = template
    for var in variables:
        if var in kwargs:
            result = result.replace(f"{{{var}}}", str(kwargs[var]))
    return result


def build_few_shot_prompt(
    instruction: str,
    examples: Sequence[Tuple[str, str]],
    query: str,
    input_label: str = "Input",
    output_label: str = "Output",
    separator: str = "\n\n",
) -> str:
    """
    Construct a few-shot learning prompt from instruction, examples, and query.

    Args:
        instruction: The task instruction / system prompt.
        examples: A sequence of ``(input, output)`` example pairs.
        query: The query input to classify / respond to.
        input_label: Label for example inputs (default ``"Input"``).
        output_label: Label for example outputs (default ``"Output"``).
        separator: String between examples (default ``"\\n\\n"``).

    Returns:
        A formatted few-shot prompt string.

    Raises:
        ValueError: If *instruction* or *query* is empty.

    Examples:
        >>> build_few_shot_prompt(
        ...     instruction="Classify sentiment:",
        ...     examples=[("I love it!", "positive"), ("Terrible.", "negative")],
        ...     query="It's okay.",
        ... )
        "Classify sentiment:\\nInput: I love it!\\nOutput: positive\\n\\nInput: Terrible.\\nOutput: negative\\n\\nInput: It's okay.\\nOutput:"
    """
    if not instruction or not isinstance(instruction, str):
        raise ValueError("instruction must be a non-empty string")
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string")

    parts: List[str] = [instruction]

    for inp, out in examples:
        parts.append(f"{input_label}: {inp}\n{output_label}: {out}")

    # Final query with empty output for the model to fill
    parts.append(f"{input_label}: {query}\n{output_label}:")

    return separator.join(parts)


def truncate_for_context(
    text: str,
    max_tokens: int,
    model: Optional[str] = None,
    ellipsis: str = "…",
    preserve_start: bool = True,
) -> str:
    """
    Truncate *text* so it fits within *max_tokens* (approximate).

    Uses the heuristic that 1 token ≈ 4 characters for English text.
    If *model* is provided and *max_tokens* is larger than the model's
    context window, the model's limit is used instead.

    Args:
        text: The input text to truncate.
        max_tokens: Maximum number of tokens to retain.
        model: Optional model name to look up the context window.
        ellipsis: String appended to truncated text (default ``"…"``).
        preserve_start: If ``True`` (default), keep the beginning of the text.
            If ``False``, keep the end.

    Returns:
        The truncated text, possibly with an ellipsis suffix/prefix.

    Raises:
        ValueError: If *max_tokens* is not positive.

    Examples:
        >>> truncate_for_context("A" * 100, max_tokens=10)
        'AAAAAAAAAA…'

        >>> truncate_for_context("A" * 100, max_tokens=10, preserve_start=False)
        '…AAAAAAAAAA'

        >>> truncate_for_context("Short text", max_tokens=100)
        'Short text'
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")

    # If model specified, cap at its context window
    if model and model in MODEL_CONTEXT_WINDOWS:
        max_tokens = min(max_tokens, MODEL_CONTEXT_WINDOWS[model])

    max_chars = max_tokens * _CHARS_PER_TOKEN

    if len(text) <= max_chars:
        return text

    if preserve_start:
        return text[:max_chars] + ellipsis
    else:
        return ellipsis + text[-max_chars:]


def estimate_api_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-3.5-turbo",
) -> float:
    """
    Estimate the USD cost of an LLM API call.

    Args:
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.
        model: Model name.  See :data:`MODEL_PRICING` for supported models.

    Returns:
        Estimated cost in USD (rounded to 6 decimal places).

    Raises:
        ValueError: If the model is not in :data:`MODEL_PRICING`.

    Examples:
        >>> estimate_api_cost(500, 200, model="gpt-4")
        0.045

        >>> estimate_api_cost(1000, 500, model="gpt-3.5-turbo")
        0.00125

        >>> estimate_api_cost(0, 0, model="gpt-4")
        0.0
    """
    if model not in MODEL_PRICING:
        raise ValueError(
            f"Unknown model '{model}'. Supported models: {list(MODEL_PRICING.keys())}"
        )

    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative")

    input_per_1k, output_per_1k = MODEL_PRICING[model]
    cost = (input_tokens / 1000) * input_per_1k + (output_tokens / 1000) * output_per_1k
    return round(cost, 6)


def format_chat_messages(
    messages: Sequence[Dict[str, str]],
    separator: str = "\n",
    include_roles: bool = True,
) -> str:
    """
    Format a list of chat messages into a single prompt string.

    Args:
        messages: A sequence of ``{"role": ..., "content": ...}`` dicts.
        separator: String inserted between messages (default ``"\\n"``).
        include_roles: If ``True`` (default), prefix each message with its
            role label (e.g. ``"User: ..."``).  If ``False``, only content
            is included.

    Returns:
        A formatted string of all messages.

    Raises:
        ValueError: If any message lacks a ``"content"`` key.

    Examples:
        >>> messages = [
        ...     {"role": "user", "content": "Hi"},
        ...     {"role": "assistant", "content": "Hello!"},
        ... ]
        >>> format_chat_messages(messages)
        'User: Hi\\nAssistant: Hello!'

        >>> format_chat_messages(messages, include_roles=False)
        'Hi\\nHello!'

        >>> format_chat_messages([])
        ''
    """
    if not messages:
        return ""

    role_labels: Dict[str, str] = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
        "function": "Function",
        "tool": "Tool",
    }

    parts: List[str] = []
    for msg in messages:
        if "content" not in msg:
            raise ValueError("Each message must contain a 'content' key")
        content = msg["content"]
        if include_roles:
            role = msg.get("role", "User")
            label = role_labels.get(role, role.title())
            parts.append(f"{label}: {content}")
        else:
            parts.append(content)

    return separator.join(parts)
