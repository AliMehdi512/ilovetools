"""
Comprehensive pytest suite for ilovetools.ai.context_manager

Covers all public functions and classes, including edge cases,
exceptions, token-budget allocation, message trimming, conversation
buffering, summary buffering, and the full ContextWindow manager.
"""

import pytest
from ilovetools.ai.context_manager import (
    ContextWindow,
    ConversationBuffer,
    ConversationSummaryBuffer,
    trim_messages,
    summarize_history,
    allocate_token_budget,
    count_message_tokens,
    estimate_tokens,
)


# ===========================================================================
#  estimate_tokens
# ===========================================================================

class TestEstimateTokens:
    def test_basic_string(self):
        assert estimate_tokens("Hello, world!") == 4

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_none_input(self):
        assert estimate_tokens(None) == 0  # type: ignore[arg-type]

    def test_long_string(self):
        assert estimate_tokens("a" * 100) == 25

    def test_single_char(self):
        assert estimate_tokens("x") == 1

    def test_non_string_input(self):
        assert estimate_tokens(12345) == 0  # type: ignore[arg-type]


# ===========================================================================
#  count_message_tokens
# ===========================================================================

class TestCountMessageTokens:
    def test_plain_string(self):
        assert count_message_tokens("Hello, world!") == 4

    def test_single_dict(self):
        result = count_message_tokens({"role": "user", "content": "Hello, world!"})
        # "user" = 1 token, "Hello, world!" = 4 tokens, +1 overhead = 6
        assert result == 6

    def test_list_of_dicts(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = count_message_tokens(msgs)
        assert result > 0

    def test_empty_list(self):
        assert count_message_tokens([]) == 0

    def test_dict_missing_keys(self):
        result = count_message_tokens({})
        # Empty role + empty content + 1 overhead = 1
        assert result == 1

    def test_different_model_overhead(self):
        msg = {"role": "user", "content": "Hi"}
        gpt_result = count_message_tokens(msg, model="gpt-4")
        claude_result = count_message_tokens(msg, model="claude-3")
        # Claude has overhead=2 vs gpt overhead=1
        assert claude_result == gpt_result + 1

    def test_unknown_model(self):
        msg = {"role": "user", "content": "Hi"}
        result = count_message_tokens(msg, model="unknown-model")
        assert result > 0

    def test_non_supported_type(self):
        assert count_message_tokens(12345) == 0  # type: ignore[arg-type]


# ===========================================================================
#  allocate_token_budget
# ===========================================================================

class TestAllocateTokenBudget:
    def test_default_ratios(self):
        result = allocate_token_budget(4000)
        assert result["system"] == 600
        assert result["history"] == 2200
        assert result["current"] == 1200
        assert sum(result.values()) == 4000

    def test_custom_ratios(self):
        result = allocate_token_budget(8000, system_ratio=0.1, history_ratio=0.6)
        assert result["system"] == 800
        assert result["history"] == 4800
        assert result["current"] == 2400
        assert sum(result.values()) == 8000

    def test_sum_always_equals_total(self):
        for total in [1000, 4096, 128000]:
            result = allocate_token_budget(total)
            assert sum(result.values()) == total

    def test_min_system_enforced(self):
        result = allocate_token_budget(500, system_ratio=0.05, history_ratio=0.85)
        assert result["system"] >= 100

    def test_min_current_enforced(self):
        result = allocate_token_budget(500, system_ratio=0.85, history_ratio=0.10)
        assert result["current"] >= 100

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            allocate_token_budget(0)

    def test_negative_total_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            allocate_token_budget(-100)

    def test_system_ratio_too_high(self):
        with pytest.raises(ValueError, match="system_ratio"):
            allocate_token_budget(1000, system_ratio=1.5)

    def test_history_ratio_too_high(self):
        with pytest.raises(ValueError, match="history_ratio"):
            allocate_token_budget(1000, history_ratio=1.5)

    def test_ratios_sum_exceeds_one(self):
        with pytest.raises(ValueError, match="must not exceed 1"):
            allocate_token_budget(1000, system_ratio=0.6, history_ratio=0.6)

    def test_zero_ratios(self):
        result = allocate_token_budget(1000, system_ratio=0, history_ratio=0)
        assert result["system"] == 100  # min_system
        assert result["current"] == 900  # everything else after redistribution
        assert sum(result.values()) == 1000

    def test_extreme_ratio(self):
        result = allocate_token_budget(1000, system_ratio=0.5, history_ratio=0.5)
        assert sum(result.values()) == 1000
        assert result["system"] == 500
        assert result["history"] == 400
        assert result["current"] >= 100  # min_current enforced


# ===========================================================================
#  trim_messages
# ===========================================================================

class TestTrimMessages:
    def test_no_trimming_needed(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = trim_messages(msgs, max_tokens=1000)
        assert result == msgs

    def test_empty_list(self):
        assert trim_messages([], max_tokens=100) == []

    def test_zero_max_tokens_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            trim_messages([{"role": "user", "content": "Hi"}], max_tokens=0)

    def test_negative_max_tokens_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            trim_messages([{"role": "user", "content": "Hi"}], max_tokens=-5)

    def test_trimming_removes_oldest(self):
        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First message that is long enough to be trimmed"},
            {"role": "assistant", "content": "Response to first message"},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": "Response to second"},
        ]
        result = trim_messages(msgs, max_tokens=30)
        assert len(result) < len(msgs)
        # System should be preserved
        assert result[0]["role"] == "system"
        # Last message should be preserved (most recent)
        assert result[-1]["content"] == "Response to second"

    def test_preserve_system(self):
        msgs = [
            {"role": "system", "content": "Important system prompt"},
            {"role": "user", "content": "x" * 200},
            {"role": "assistant", "content": "y" * 200},
            {"role": "user", "content": "z" * 200},
        ]
        result = trim_messages(msgs, max_tokens=50, preserve_system=True)
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Important system prompt"

    def test_no_preserve_system(self):
        msgs = [
            {"role": "system", "content": "x" * 200},
            {"role": "user", "content": "short"},
        ]
        result = trim_messages(msgs, max_tokens=5, preserve_system=False)
        # System message should be trimmed if not preserved
        assert len(result) <= 1

    def test_preserve_first_and_last(self):
        msgs = [
            {"role": "user", "content": "FIRST"},
            {"role": "user", "content": "x" * 200},
            {"role": "user", "content": "y" * 200},
            {"role": "user", "content": "LAST"},
        ]
        result = trim_messages(msgs, max_tokens=20, preserve_system=False, preserve_first=1, preserve_last=1)
        if len(result) < len(msgs):
            assert result[0]["content"] == "FIRST"
            assert result[-1]["content"] == "LAST"

    def test_returns_new_list(self):
        msgs = [{"role": "user", "content": "Hi"}]
        result = trim_messages(msgs, max_tokens=100)
        assert result is not msgs

    def test_all_system_messages_preserved(self):
        msgs = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
            {"role": "user", "content": "x" * 200},
        ]
        result = trim_messages(msgs, max_tokens=30, preserve_system=True)
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 2


# ===========================================================================
#  summarize_history
# ===========================================================================

class TestSummarizeHistory:
    def test_basic_summary(self):
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        result = summarize_history(msgs)
        assert result["role"] == "system"
        assert "2+2" in result["content"]
        assert "Conversation summary" in result["content"]

    def test_empty_messages(self):
        result = summarize_history([])
        assert result == {"role": "system", "content": ""}

    def test_custom_summarizer(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        custom = lambda ms: f"{len(ms)} messages"
        result = summarize_history(msgs, summarizer=custom)
        assert result["content"] == "2 messages"

    def test_max_messages_limit(self):
        msgs = [{"role": "user", "content": f"Message {i}"} for i in range(20)]
        result = summarize_history(msgs, max_messages=5)
        assert "Message 0" in result["content"]
        assert "Message 4" in result["content"]
        assert "Message 5" not in result["content"]
        assert "15 earlier messages omitted" in result["content"]

    def test_custom_system_role(self):
        msgs = [{"role": "user", "content": "Hi"}]
        result = summarize_history(msgs, system_role="assistant")
        assert result["role"] == "assistant"

    def test_long_message_truncated(self):
        long_content = "A" * 500
        msgs = [{"role": "user", "content": long_content}]
        result = summarize_history(msgs)
        assert len(result["content"]) < 500 + 100  # summary header + truncated content


# ===========================================================================
#  ConversationBuffer
# ===========================================================================

class TestConversationBuffer:
    def test_basic_add_and_get(self):
        buf = ConversationBuffer(max_tokens=500)
        buf.add_user_message("Hello!")
        buf.add_assistant_message("Hi there!")
        msgs = buf.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_add_message(self):
        buf = ConversationBuffer(max_tokens=500)
        buf.add_message("system", "Be helpful.")
        msgs = buf.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_add_system_message(self):
        buf = ConversationBuffer(max_tokens=500)
        buf.add_system_message("System prompt")
        msgs = buf.get_messages()
        assert msgs[0]["role"] == "system"

    def test_add_many(self):
        buf = ConversationBuffer(max_tokens=500)
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        buf.add_many(msgs)
        assert len(buf) == 2

    def test_clear(self):
        buf = ConversationBuffer(max_tokens=500)
        buf.add_user_message("Hello!")
        buf.clear()
        assert len(buf) == 0
        assert buf.get_messages() == []

    def test_token_count(self):
        buf = ConversationBuffer(max_tokens=500)
        buf.add_user_message("Hello, world!")
        assert buf.token_count() > 0

    def test_eviction_removes_oldest(self):
        buf = ConversationBuffer(max_tokens=20)
        buf.add_user_message("This is a very long message that takes many tokens")
        buf.add_user_message("Short")
        buf.add_user_message("Also short")
        msgs = buf.get_messages()
        assert buf.token_count() <= 20
        long_msg = [m for m in msgs if "very long message" in m["content"]]
        assert len(long_msg) == 0

    def test_system_messages_not_evicted(self):
        buf = ConversationBuffer(max_tokens=30)
        buf.add_system_message("Important system prompt that should be kept")
        buf.add_user_message("x" * 100)
        buf.add_user_message("y" * 100)
        msgs = buf.get_messages()
        system_msgs = [m for m in msgs if m["role"] == "system"]
        assert len(system_msgs) == 1

    def test_max_tokens_property(self):
        buf = ConversationBuffer(max_tokens=777)
        assert buf.max_tokens == 777

    def test_model_property(self):
        buf = ConversationBuffer(max_tokens=100, model="gpt-4")
        assert buf.model == "gpt-4"

    def test_repr(self):
        buf = ConversationBuffer(max_tokens=100)
        buf.add_user_message("Hi")
        r = repr(buf)
        assert "ConversationBuffer" in r
        assert "max_tokens=100" in r

    def test_len(self):
        buf = ConversationBuffer(max_tokens=500)
        buf.add_user_message("Hi")
        buf.add_assistant_message("Hello")
        assert len(buf) == 2

    def test_zero_max_tokens_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            ConversationBuffer(max_tokens=0)

    def test_negative_max_tokens_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            ConversationBuffer(max_tokens=-10)

    def test_get_messages_returns_copy(self):
        buf = ConversationBuffer(max_tokens=500)
        buf.add_user_message("Hi")
        msgs1 = buf.get_messages()
        msgs1.append({"role": "user", "content": "extra"})
        msgs2 = buf.get_messages()
        assert len(msgs2) == 1


# ===========================================================================
#  ConversationSummaryBuffer
# ===========================================================================

class TestConversationSummaryBuffer:
    def test_basic_add_and_payload(self):
        buf = ConversationSummaryBuffer(max_tokens=200)
        buf.add_user_message("Hello!")
        buf.add_assistant_message("Hi!")
        payload = buf.get_payload()
        assert isinstance(payload, list)
        assert len(payload) >= 2

    def test_add_many(self):
        buf = ConversationSummaryBuffer(max_tokens=500)
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        buf.add_many(msgs)
        assert len(buf) == 2

    def test_clear(self):
        buf = ConversationSummaryBuffer(max_tokens=200)
        buf.add_user_message("Hello!")
        buf.clear()
        assert len(buf) == 0
        assert buf.get_summary() == ""

    def test_get_summary_empty(self):
        buf = ConversationSummaryBuffer(max_tokens=200)
        assert buf.get_summary() == ""

    def test_get_summary_after_summarization(self):
        buf = ConversationSummaryBuffer(max_tokens=30)
        for i in range(10):
            buf.add_user_message(f"Message number {i} with some content")
        assert buf.get_summary() != ""

    def test_get_messages(self):
        buf = ConversationSummaryBuffer(max_tokens=500)
        buf.add_user_message("Hi")
        msgs = buf.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_token_count(self):
        buf = ConversationSummaryBuffer(max_tokens=200)
        buf.add_user_message("Hello!")
        assert buf.token_count() > 0

    def test_payload_fits_budget(self):
        buf = ConversationSummaryBuffer(max_tokens=100)
        for i in range(20):
            buf.add_user_message(f"Message {i} with enough text to trigger summarization")
        payload = buf.get_payload()
        total = count_message_tokens(payload)
        assert total <= 100

    def test_custom_summarizer(self):
        custom = lambda ms: f"SUMMARY: {len(ms)} messages"
        buf = ConversationSummaryBuffer(max_tokens=30, summarizer=custom)
        for i in range(5):
            buf.add_user_message(f"Message {i} with content here")
        summary = buf.get_summary()
        if summary:
            assert "SUMMARY:" in summary

    def test_max_tokens_property(self):
        buf = ConversationSummaryBuffer(max_tokens=333)
        assert buf.max_tokens == 333

    def test_repr(self):
        buf = ConversationSummaryBuffer(max_tokens=100)
        buf.add_user_message("Hi")
        r = repr(buf)
        assert "ConversationSummaryBuffer" in r
        assert "max_tokens=100" in r

    def test_zero_max_tokens_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            ConversationSummaryBuffer(max_tokens=0)

    def test_negative_max_tokens_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            ConversationSummaryBuffer(max_tokens=-5)

    def test_add_assistant_message(self):
        buf = ConversationSummaryBuffer(max_tokens=200)
        buf.add_assistant_message("Response")
        msgs = buf.get_messages()
        assert msgs[0]["role"] == "assistant"

    def test_summary_prepended_in_payload(self):
        buf = ConversationSummaryBuffer(max_tokens=30)
        for i in range(10):
            buf.add_user_message(f"Message {i} with some content here")
        payload = buf.get_payload()
        if buf.get_summary():
            assert payload[0]["role"] == "system"
            assert "Conversation summary" in payload[0]["content"] or "SUMMARY" in payload[0]["content"]


# ===========================================================================
#  ContextWindow
# ===========================================================================

class TestContextWindow:
    def test_basic_usage(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=500)
        cw.set_system_prompt("You are a helpful assistant.")
        cw.add_user_message("What is 2+2?")
        cw.add_assistant_message("4")
        cw.add_user_message("And 3+3?")
        payload = cw.build_payload()
        assert isinstance(payload, list)
        assert payload[0]["role"] == "system"
        assert payload[0]["content"] == "You are a helpful assistant."

    def test_clear_history(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=500)
        cw.set_system_prompt("Test")
        cw.add_user_message("Hello")
        cw.clear_history()
        assert len(cw) == 0
        assert cw.get_system_prompt() == "Test"

    def test_get_system_prompt(self):
        cw = ContextWindow(model="gpt-4")
        cw.set_system_prompt("My prompt")
        assert cw.get_system_prompt() == "My prompt"

    def test_get_history(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=500)
        cw.add_user_message("Hi")
        cw.add_assistant_message("Hello")
        history = cw.get_history()
        assert len(history) == 2

    def test_add_message(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=500)
        cw.add_message("custom_role", "Custom content")
        history = cw.get_history()
        assert history[0]["role"] == "custom_role"

    def test_add_many(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=500)
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        cw.add_many(msgs)
        assert len(cw) == 2

    def test_token_count(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=500)
        cw.set_system_prompt("Test prompt")
        cw.add_user_message("Hello")
        assert cw.token_count() > 0

    def test_remaining_tokens(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=500)
        cw.set_system_prompt("Test")
        cw.add_user_message("Hi")
        remaining = cw.remaining_tokens()
        assert remaining > 0
        assert remaining < cw.input_budget

    def test_model_property(self):
        cw = ContextWindow(model="gpt-4o")
        assert cw.model == "gpt-4o"

    def test_context_window_property(self):
        cw = ContextWindow(model="gpt-4")
        assert cw.context_window == 8192

    def test_context_window_override(self):
        cw = ContextWindow(model="gpt-4", context_window=16000)
        assert cw.context_window == 16000

    def test_reserve_tokens_property(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=1000)
        assert cw.reserve_tokens == 1000

    def test_input_budget_property(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=500)
        assert cw.input_budget == 8192 - 500

    def test_unknown_model_defaults(self):
        cw = ContextWindow(model="unknown-model-xyz")
        assert cw.context_window == 4096

    def test_repr(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=500)
        cw.add_user_message("Hi")
        r = repr(cw)
        assert "ContextWindow" in r
        assert "gpt-4" in r

    def test_len(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=500)
        cw.add_user_message("Hi")
        cw.add_assistant_message("Hello")
        assert len(cw) == 2

    def test_negative_reserve_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ContextWindow(model="gpt-4", reserve_tokens=-100)

    def test_reserve_exceeds_window_raises(self):
        with pytest.raises(ValueError, match="smaller than context_window"):
            ContextWindow(model="gpt-4", reserve_tokens=8192)

    def test_invalid_system_ratio(self):
        with pytest.raises(ValueError, match="system_ratio"):
            ContextWindow(model="gpt-4", system_ratio=1.5)

    def test_invalid_history_ratio(self):
        with pytest.raises(ValueError, match="history_ratio"):
            ContextWindow(model="gpt-4", history_ratio=1.5)

    def test_ratios_exceed_one(self):
        with pytest.raises(ValueError, match="must not exceed 1"):
            ContextWindow(model="gpt-4", system_ratio=0.6, history_ratio=0.6)

    def test_payload_fits_budget(self):
        cw = ContextWindow(model="gpt-3.5-turbo", reserve_tokens=100)
        cw.set_system_prompt("You are helpful.")
        for i in range(50):
            cw.add_user_message(f"Message {i} " * 20)
        payload = cw.build_payload()
        total = count_message_tokens(payload)
        assert total <= cw.input_budget

    def test_no_system_prompt(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=500)
        cw.add_user_message("Hi")
        payload = cw.build_payload()
        assert payload[0]["role"] == "user"

    def test_reserve_equals_window_raises(self):
        with pytest.raises(ValueError, match="smaller than context_window"):
            ContextWindow(model="gpt-4", reserve_tokens=8192)

    def test_zero_reserve(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=0)
        assert cw.input_budget == 8192

    def test_add_assistant_message(self):
        cw = ContextWindow(model="gpt-4", reserve_tokens=500)
        cw.add_assistant_message("Response")
        history = cw.get_history()
        assert history[0]["role"] == "assistant"

    def test_set_system_prompt_overwrites(self):
        cw = ContextWindow(model="gpt-4")
        cw.set_system_prompt("First")
        cw.set_system_prompt("Second")
        assert cw.get_system_prompt() == "Second"
