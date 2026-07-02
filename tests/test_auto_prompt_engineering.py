"""
Comprehensive pytest suite for ilovetools.ai.prompt_engineering

Covers all public functions and classes, edge cases, exception paths,
and typical usage scenarios.
"""

import pytest
from ilovetools.ai.prompt_engineering import (
    PromptBuilder,
    PromptTemplate,
    build_few_shot_prompt,
    extract_variables,
    fill_template,
    truncate_for_context,
    estimate_api_cost,
    format_chat_messages,
    MODEL_CONTEXT_WINDOWS,
    MODEL_PRICING,
)


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    """Tests for the PromptBuilder fluent API."""

    def test_system_user_chain(self):
        builder = (
            PromptBuilder()
            .system("You are a helpful assistant.")
            .user("What is 2+2?")
        )
        msgs = builder.to_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert msgs[1] == {"role": "user", "content": "What is 2+2?"}

    def test_to_string(self):
        builder = (
            PromptBuilder()
            .system("Be concise.")
            .user("Hi")
        )
        result = builder.to_string()
        assert "System: Be concise." in result
        assert "User: Hi" in result
        assert "\n" in result

    def test_to_string_custom_separator(self):
        builder = PromptBuilder().user("Hello")
        assert builder.to_string(separator=" | ") == "User: Hello"

    def test_assistant_and_function(self):
        builder = (
            PromptBuilder()
            .user("Call the API")
            .assistant("Sure")
            .function("result_data", name="get_weather")
        )
        msgs = builder.to_messages()
        assert msgs[2]["role"] == "function"
        assert msgs[2]["name"] == "get_weather"

    def test_tool_message(self):
        builder = PromptBuilder().tool("tool_output", tool_call_id="call_123")
        msgs = builder.to_messages()
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_123"

    def test_add_arbitrary_role(self):
        builder = PromptBuilder().add("custom_role", "custom content")
        msgs = builder.to_messages()
        assert msgs[0]["role"] == "custom_role"
        assert msgs[0]["content"] == "custom content"

    def test_add_empty_role_raises(self):
        with pytest.raises(ValueError, match="role must be a non-empty string"):
            PromptBuilder().add("", "content")

    def test_non_string_content_raises(self):
        with pytest.raises(TypeError):
            PromptBuilder().system(123)
        with pytest.raises(TypeError):
            PromptBuilder().user(None)
        with pytest.raises(TypeError):
            PromptBuilder().assistant([])

    def test_token_count_positive(self):
        builder = PromptBuilder().system("You are helpful.").user("Hello world!")
        assert builder.token_count() > 0

    def test_token_count_empty(self):
        assert PromptBuilder().token_count() == 0

    def test_clear(self):
        builder = PromptBuilder().system("test").user("test")
        assert len(builder) == 2
        builder.clear()
        assert len(builder) == 0
        assert builder.to_messages() == []

    def test_len(self):
        builder = PromptBuilder().system("a").user("b").assistant("c")
        assert len(builder) == 3

    def test_repr(self):
        builder = PromptBuilder().system("a").user("b")
        r = repr(builder)
        assert "PromptBuilder" in r
        assert "2" in r

    def test_to_messages_returns_copy(self):
        builder = PromptBuilder().user("hello")
        msgs = builder.to_messages()
        msgs.append({"role": "user", "content": "extra"})
        # Original should be unaffected
        assert len(builder) == 1

    def test_fluent_returns_self(self):
        builder = PromptBuilder()
        assert builder.system("a") is builder
        assert builder.user("b") is builder
        assert builder.assistant("c") is builder


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------

class TestPromptTemplate:
    """Tests for the PromptTemplate class."""

    def test_extract_variables(self):
        tpl = PromptTemplate("Hello {name}, your code is {code}")
        assert tpl.variables == ["name", "code"]

    def test_no_variables(self):
        tpl = PromptTemplate("No variables here")
        assert tpl.variables == []

    def test_duplicate_variables(self):
        tpl = PromptTemplate("{a} and {a} and {b}")
        assert tpl.variables == ["a", "b"]

    def test_render_success(self):
        tpl = PromptTemplate("Translate {source} to {target}: {text}")
        result = tpl.render(source="English", target="French", text="Hello")
        assert result == "Translate English to French: Hello"

    def test_render_missing_variable_raises(self):
        tpl = PromptTemplate("Hello {name} and {place}")
        with pytest.raises(KeyError, match="Missing required variables"):
            tpl.render(name="World")

    def test_render_unexpected_variable_raises(self):
        tpl = PromptTemplate("Hello {name}")
        with pytest.raises(KeyError, match="Unexpected variables"):
            tpl.render(name="World", extra="nope")

    def test_render_no_validation(self):
        tpl = PromptTemplate("Hello {name}", validate_on_render=False)
        # Missing variable -> left as literal
        result = tpl.render()
        assert result == "Hello {name}"

    def test_render_no_variables(self):
        tpl = PromptTemplate("Static text")
        assert tpl.render() == "Static text"

    def test_non_string_template_raises(self):
        with pytest.raises(TypeError):
            PromptTemplate(123)

    def test_repr(self):
        tpl = PromptTemplate("Hello {name}")
        assert "PromptTemplate" in repr(tpl)
        assert "name" in repr(tpl)


# ---------------------------------------------------------------------------
# extract_variables
# ---------------------------------------------------------------------------

class TestExtractVariables:
    def test_basic(self):
        assert extract_variables("Hello {name}, your code is {code}") == ["name", "code"]

    def test_none(self):
        assert extract_variables("No variables") == []

    def test_empty_string(self):
        assert extract_variables("") == []

    def test_non_string(self):
        assert extract_variables(123) == []  # type: ignore

    def test_duplicates(self):
        assert extract_variables("{a}{b}{a}") == ["a", "b"]

    def test_underscore_and_digits(self):
        assert extract_variables("{var_1} and {var_2}") == ["var_1", "var_2"]


# ---------------------------------------------------------------------------
# fill_template
# ---------------------------------------------------------------------------

class TestFillTemplate:
    def test_basic(self):
        assert fill_template("Hello {name}!", name="World") == "Hello World!"

    def test_partial_fill(self):
        result = fill_template("Hello {name} and {other}!", name="World")
        assert result == "Hello World and {other}!"

    def test_strict_missing_raises(self):
        with pytest.raises(KeyError, match="Missing required variables"):
            fill_template("Hello {name}!", strict=True)

    def test_strict_all_provided(self):
        assert fill_template("Hello {name}!", strict=True, name="World") == "Hello World!"

    def test_no_variables(self):
        assert fill_template("No vars") == "No vars"

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            fill_template(123)  # type: ignore

    def test_integer_value(self):
        assert fill_template("Count: {n}", n=42) == "Count: 42"


# ---------------------------------------------------------------------------
# build_few_shot_prompt
# ---------------------------------------------------------------------------

class TestBuildFewShotPrompt:
    def test_basic(self):
        result = build_few_shot_prompt(
            instruction="Classify sentiment:",
            examples=[("I love it!", "positive"), ("Terrible.", "negative")],
            query="It's okay.",
        )
        assert "Classify sentiment:" in result
        assert "Input: I love it!" in result
        assert "Output: positive" in result
        assert "Input: Terrible." in result
        assert "Output: negative" in result
        assert "Input: It's okay." in result
        assert result.endswith("Output:")

    def test_custom_labels(self):
        result = build_few_shot_prompt(
            instruction="Task:",
            examples=[("q1", "a1")],
            query="q2",
            input_label="Question",
            output_label="Answer",
        )
        assert "Question: q1" in result
        assert "Answer: a1" in result
        assert "Question: q2" in result

    def test_no_examples(self):
        result = build_few_shot_prompt(
            instruction="Do something:",
            examples=[],
            query="test",
        )
        assert "Do something:" in result
        assert "Input: test" in result
        assert result.endswith("Output:")

    def test_empty_instruction_raises(self):
        with pytest.raises(ValueError, match="instruction must be a non-empty string"):
            build_few_shot_prompt("", [], "query")

    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            build_few_shot_prompt("instruction", [], "")

    def test_custom_separator(self):
        result = build_few_shot_prompt(
            instruction="Task:",
            examples=[("a", "b")],
            query="c",
            separator="---",
        )
        assert "---" in result


# ---------------------------------------------------------------------------
# truncate_for_context
# ---------------------------------------------------------------------------

class TestTruncateForContext:
    def test_no_truncation_needed(self):
        assert truncate_for_context("Short text", max_tokens=100) == "Short text"

    def test_truncate_preserve_start(self):
        result = truncate_for_context("A" * 100, max_tokens=10)
        assert result.endswith("…")
        assert len(result) == 41  # 10*4 + 1 (ellipsis)

    def test_truncate_preserve_end(self):
        result = truncate_for_context("A" * 100, max_tokens=10, preserve_start=False)
        assert result.startswith("…")
        assert len(result) == 41

    def test_custom_ellipsis(self):
        result = truncate_for_context("A" * 100, max_tokens=10, ellipsis="...")
        assert result.endswith("...")
        assert len(result) == 43  # 40 + 3

    def test_model_context_window_cap(self):
        # gpt-3.5-turbo has 4096 context window
        result = truncate_for_context("A" * 100000, max_tokens=999999, model="gpt-3.5-turbo")
        # Should be capped at 4096 tokens = 16384 chars + ellipsis
        assert len(result) == 16385  # 4096*4 + 1

    def test_zero_max_tokens_raises(self):
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            truncate_for_context("text", max_tokens=0)

    def test_negative_max_tokens_raises(self):
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            truncate_for_context("text", max_tokens=-5)

    def test_non_string_text_raises(self):
        with pytest.raises(TypeError):
            truncate_for_context(123, max_tokens=10)  # type: ignore


# ---------------------------------------------------------------------------
# estimate_api_cost
# ---------------------------------------------------------------------------

class TestEstimateApiCost:
    def test_gpt4(self):
        cost = estimate_api_cost(500, 200, model="gpt-4")
        # 500/1000 * 0.03 + 200/1000 * 0.06 = 0.015 + 0.012 = 0.027
        assert cost == 0.027

    def test_gpt35(self):
        cost = estimate_api_cost(1000, 500, model="gpt-3.5-turbo")
        # 1000/1000 * 0.0005 + 500/1000 * 0.0015 = 0.0005 + 0.00075 = 0.00125
        assert cost == 0.00125

    def test_zero_tokens(self):
        assert estimate_api_cost(0, 0, model="gpt-4") == 0.0

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            estimate_api_cost(100, 50, model="nonexistent-model")

    def test_negative_tokens_raises(self):
        with pytest.raises(ValueError, match="Token counts must be non-negative"):
            estimate_api_cost(-1, 100, model="gpt-4")

    def test_all_models_have_pricing(self):
        """Every model in MODEL_CONTEXT_WINDOWS should also have pricing."""
        for model in MODEL_CONTEXT_WINDOWS:
            assert model in MODEL_PRICING, f"Missing pricing for {model}"


# ---------------------------------------------------------------------------
# format_chat_messages
# ---------------------------------------------------------------------------

class TestFormatChatMessages:
    def test_basic(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = format_chat_messages(messages)
        assert result == "User: Hi\nAssistant: Hello!"

    def test_without_roles(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = format_chat_messages(messages, include_roles=False)
        assert result == "Hi\nHello!"

    def test_empty_messages(self):
        assert format_chat_messages([]) == ""

    def test_custom_separator(self):
        messages = [{"role": "user", "content": "Hi"}]
        assert format_chat_messages(messages, separator=" | ") == "User: Hi"

    def test_system_role(self):
        messages = [{"role": "system", "content": "Be helpful."}]
        assert format_chat_messages(messages) == "System: Be helpful."

    def test_tool_role(self):
        messages = [{"role": "tool", "content": "result"}]
        assert format_chat_messages(messages) == "Tool: result"

    def test_unknown_role(self):
        messages = [{"role": "narrator", "content": "Once upon a time..."}]
        result = format_chat_messages(messages)
        assert result == "Narrator: Once upon a time..."

    def test_missing_content_raises(self):
        messages = [{"role": "user"}]  # no content key
        with pytest.raises(ValueError, match="must contain a 'content' key"):
            format_chat_messages(messages)

    def test_custom_separator_with_multiple(self):
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        result = format_chat_messages(messages, separator=" || ")
        assert result == "User: A || Assistant: B || User: C"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration tests combining multiple utilities."""

    def test_prompt_builder_with_template(self):
        """Use PromptTemplate inside a PromptBuilder."""
        tpl = PromptTemplate("Summarise the following: {text}")
        builder = (
            PromptBuilder()
            .system("You are a summarisation assistant.")
            .user(tpl.render(text="Long article text here..."))
        )
        msgs = builder.to_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[1]["content"] == "Summarise the following: Long article text here..."

    def test_few_shot_with_truncation(self):
        """Build a few-shot prompt and truncate it for context."""
        long_examples = [("A" * 200, "B" * 200)] * 5
        prompt = build_few_shot_prompt(
            instruction="Classify:",
            examples=long_examples,
            query="test query",
        )
        truncated = truncate_for_context(prompt, max_tokens=50)
        assert truncated.endswith("…")
        assert len(truncated) <= 201  # 50*4 + 1

    def test_cost_estimation_with_builder(self):
        """Estimate cost of a PromptBuilder's messages."""
        builder = (
            PromptBuilder()
            .system("You are a helpful coding assistant. " * 20)
            .user("Write a Python function to sort a list. " * 10)
        )
        tokens = builder.token_count()
        cost = estimate_api_cost(tokens, 500, model="gpt-4")
        assert cost > 0

    def test_template_with_fill_template(self):
        """PromptTemplate and fill_template should produce consistent results."""
        template_str = "Hello {name}, welcome to {place}!"
        tpl = PromptTemplate(template_str)
        a = tpl.render(name="Alice", place="Wonderland")
        b = fill_template(template_str, name="Alice", place="Wonderland")
        assert a == b

    def test_full_workflow(self):
        """Complete workflow: template -> builder -> format -> cost."""
        tpl = PromptTemplate(
            "Analyse the following code for bugs:\n{code}"
        )
        rendered = tpl.render(code="def foo(): pass")
        builder = (
            PromptBuilder()
            .system("You are a code review assistant.")
            .user(rendered)
        )
        # Convert to flat string
        flat = builder.to_string()
        assert "code review assistant" in flat
        assert "def foo(): pass" in flat

        # Estimate cost
        tokens = builder.token_count()
        cost = estimate_api_cost(tokens, 200, model="gpt-4o-mini")
        assert cost > 0
        assert isinstance(cost, float)
