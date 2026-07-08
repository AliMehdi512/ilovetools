"""
Tests for ilovetools.ai.tool_registry — LLM tool-calling registry.

Covers:
- Tool creation and schema generation
- Type-hint to JSON-schema mapping
- Docstring parsing (summary + args)
- ToolRegistry registration, unregistration, listing
- Schema generation for OpenAI / Anthropic / Gemini providers
- Tool execution (direct, safe, batch)
- ToolResult serialisation
- @tool decorator
- Edge cases: missing args, unknown tools, execution errors
"""

import json
import pytest
from typing import List, Optional, Dict, Union

from ilovetools.ai.tool_registry import (
    ToolRegistry,
    Tool,
    tool,
    ToolError,
    ToolResult,
    _python_type_to_json_schema,
    _parse_docstring_args,
    _parse_docstring_summary,
)


# ---------------------------------------------------------------------------
# _python_type_to_json_schema
# ---------------------------------------------------------------------------

class TestPythonTypeToSchema:
    def test_str(self):
        assert _python_type_to_json_schema(str) == {"type": "string"}

    def test_int(self):
        assert _python_type_to_json_schema(int) == {"type": "integer"}

    def test_float(self):
        assert _python_type_to_json_schema(float) == {"type": "number"}

    def test_bool(self):
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_list(self):
        assert _python_type_to_json_schema(list) == {"type": "array"}

    def test_dict(self):
        assert _python_type_to_json_schema(dict) == {"type": "object"}

    def test_list_of_int(self):
        schema = _python_type_to_json_schema(List[int])
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "integer"

    def test_list_of_str(self):
        schema = _python_type_to_json_schema(List[str])
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_dict_str_str(self):
        schema = _python_type_to_json_schema(Dict[str, str])
        assert schema["type"] == "object"
        assert schema["additionalProperties"]["type"] == "string"

    def test_optional_str(self):
        schema = _python_type_to_json_schema(Optional[str])
        assert "anyOf" in schema
        assert {"type": "string"} in schema["anyOf"]
        assert {"type": "null"} in schema["anyOf"]

    def test_union_str_int(self):
        schema = _python_type_to_json_schema(Union[str, int])
        assert "anyOf" in schema
        assert len(schema["anyOf"]) == 2

    def test_empty_annotation(self):
        import inspect
        schema = _python_type_to_json_schema(inspect.Parameter.empty)
        assert schema["type"] == "string"

    def test_none_annotation(self):
        schema = _python_type_to_json_schema(None)
        assert schema["type"] == "string"


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------

class TestDocstringParsing:
    def test_summary_single_line(self):
        assert _parse_docstring_summary("Hello world.") == "Hello world."

    def test_summary_multi_line(self):
        doc = "First line.\nSecond line.\n\nArgs:\n    x: foo"
        assert _parse_docstring_summary(doc) == "First line. Second line."

    def test_summary_empty(self):
        assert _parse_docstring_summary("") == ""

    def test_args_basic(self):
        doc = "Summary.\n\nArgs:\n    city: The city name.\n    units: Temp units."
        result = _parse_docstring_args(doc)
        assert result["city"] == "The city name."
        assert result["units"] == "Temp units."

    def test_args_with_type_hint(self):
        doc = "Summary.\n\nArgs:\n    city (str): The city name."
        result = _parse_docstring_args(doc)
        assert result["city"] == "The city name."

    def test_args_multiline_desc(self):
        doc = "Summary.\n\nArgs:\n    city: The city name.\n        Can be any city.\n    units: Units."
        result = _parse_docstring_args(doc)
        assert "The city name" in result["city"]
        assert "Can be any city" in result["city"]

    def test_args_none(self):
        assert _parse_docstring_args("No args here.") == {}

    def test_args_empty(self):
        assert _parse_docstring_args("") == {}

    def test_parameters_alias(self):
        doc = "Summary.\n\nParameters:\n    x: The x value."
        result = _parse_docstring_args(doc)
        assert result["x"] == "The x value."


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class TestTool:
    def test_basic_creation(self):
        def greet(name: str) -> str:
            """Greet someone.

            Args:
                name: Person's name.
            """
            return f"Hello, {name}!"

        t = Tool("greet", greet)
        assert t.name == "greet"
        assert t.description == "Greet someone."
        assert "name" in t.schema["properties"]
        assert t.schema["properties"]["name"]["type"] == "string"
        assert "name" in t.schema["required"]
        assert t.schema["properties"]["name"]["description"] == "Person's name."

    def test_with_default_param(self):
        def f(a: int, b: str = "x") -> str:
            """Do something.

            Args:
                a: First.
                b: Second.
            """
            return str(a) + b

        t = Tool("f", f)
        assert "a" in t.schema["required"]
        assert "b" not in t.schema["required"]

    def test_execute_success(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        t = Tool("add", add)
        assert t.execute({"a": 3, "b": 4}) == 7

    def test_execute_type_error(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        t = Tool("add", add)
        with pytest.raises(ToolError, match="Invalid arguments"):
            t.execute({"a": 3})  # missing b

    def test_execute_runtime_error(self):
        def fail(x: int) -> int:
            """Always fails."""
            raise ValueError("boom")

        t = Tool("fail", fail)
        with pytest.raises(ToolError, match="execution failed"):
            t.execute({"x": 1})

    def test_custom_description(self):
        def f(x: int) -> int:
            """Original docstring."""
            return x

        t = Tool("f", f, description="Custom description")
        assert t.description == "Custom description"

    def test_no_docstring(self):
        def f(x: int) -> int:
            return x

        t = Tool("f", f)
        assert t.description == ""
        assert "x" in t.schema["properties"]

    def test_repr(self):
        def f(x: int) -> int:
            """Test."""
            return x

        t = Tool("f", f)
        assert "Tool" in repr(t)
        assert "f" in repr(t)


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

class TestToolResult:
    def test_success(self):
        r = ToolResult("add", True, 42, None)
        assert r.success
        assert r.result == 42
        assert r.error is None

    def test_failure(self):
        r = ToolResult("add", False, None, "some error")
        assert not r.success
        assert r.result is None
        assert r.error == "some error"

    def test_to_dict(self):
        r = ToolResult("add", True, 42, None)
        d = r.to_dict()
        assert d["name"] == "add"
        assert d["success"] is True
        assert d["result"] == 42
        assert d["error"] is None

    def test_to_json(self):
        r = ToolResult("add", True, 42, None)
        parsed = json.loads(r.to_json())
        assert parsed["result"] == 42
        assert parsed["success"] is True

    def test_repr_success(self):
        r = ToolResult("add", True, 42, None)
        assert "add" in repr(r)
        assert "42" in repr(r)

    def test_repr_failure(self):
        r = ToolResult("add", False, None, "error")
        assert "add" in repr(r)
        assert "error" in repr(r)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_empty_registry(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        assert reg.list_names() == []

    def test_register(self):
        reg = ToolRegistry()
        def f(x: int) -> int:
            """Double."""
            return x * 2
        t = reg.register(f)
        assert t.name == "f"
        assert "f" in reg
        assert len(reg) == 1

    def test_register_with_custom_name(self):
        reg = ToolRegistry()
        def f(x: int) -> int:
            """Double."""
            return x * 2
        t = reg.register(f, name="double")
        assert t.name == "double"
        assert "double" in reg

    def test_register_with_custom_description(self):
        reg = ToolRegistry()
        def f(x: int) -> int:
            """Original."""
            return x
        t = reg.register(f, description="Custom desc")
        assert t.description == "Custom desc"

    def test_unregister(self):
        reg = ToolRegistry()
        def f(x: int) -> int:
            """Test."""
            return x
        reg.register(f, name="test")
        assert "test" in reg
        reg.unregister("test")
        assert "test" not in reg

    def test_unregister_not_found(self):
        reg = ToolRegistry()
        with pytest.raises(ToolError, match="not found"):
            reg.unregister("nonexistent")

    def test_list_names_sorted(self):
        reg = ToolRegistry()
        def f1(x: int) -> int:
            """F1."""
            return x
        def f2(x: int) -> int:
            """F2."""
            return x
        reg.register(f1, name="zebra")
        reg.register(f2, name="alpha")
        assert reg.list_names() == ["alpha", "zebra"]

    def test_info(self):
        reg = ToolRegistry()
        def greet(name: str) -> str:
            """Greet someone.

            Args:
                name: Person's name.
            """
            return f"Hi {name}"
        reg.register(greet)
        info = reg.info()
        assert "greet" in info
        assert "name" in info["greet"]["parameters"]
        assert "name" in info["greet"]["required"]
        assert info["greet"]["description"] == "Greet someone."

    def test_repr(self):
        reg = ToolRegistry()
        def f(x: int) -> int:
            """Test."""
            return x
        reg.register(f)
        assert "ToolRegistry" in repr(reg)
        assert "1" in repr(reg)


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------

class TestSchemaGeneration:
    def test_openai_format(self):
        reg = ToolRegistry()
        def greet(name: str) -> str:
            """Greet someone.

            Args:
                name: Person's name.
            """
            return f"Hi {name}"
        reg.register(greet)
        schemas = reg.schemas("openai")
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "greet"
        assert schemas[0]["function"]["description"] == "Greet someone."
        assert schemas[0]["function"]["parameters"]["properties"]["name"]["type"] == "string"
        assert "name" in schemas[0]["function"]["parameters"]["required"]

    def test_anthropic_format(self):
        reg = ToolRegistry()
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}"
        reg.register(greet)
        schemas = reg.schemas("anthropic")
        assert schemas[0]["name"] == "greet"
        assert schemas[0]["description"] == "Greet someone."
        assert "input_schema" in schemas[0]
        assert schemas[0]["input_schema"]["properties"]["name"]["type"] == "string"

    def test_gemini_format(self):
        reg = ToolRegistry()
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}"
        reg.register(greet)
        schemas = reg.schemas("gemini")
        assert schemas[0]["name"] == "greet"
        assert "parameters" in schemas[0]
        assert schemas[0]["parameters"]["properties"]["name"]["type"] == "string"

    def test_invalid_provider(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="Unsupported provider"):
            reg.schemas("invalid")

    def test_schemas_json(self):
        reg = ToolRegistry()
        def f(x: int) -> int:
            """Square."""
            return x ** 2
        reg.register(f)
        json_str = reg.schemas_json()
        parsed = json.loads(json_str)
        assert parsed[0]["function"]["name"] == "f"

    def test_multiple_tools(self):
        reg = ToolRegistry()
        def f1(x: int) -> int:
            """F1."""
            return x
        def f2(y: str) -> str:
            """F2."""
            return y
        reg.register(f1, name="tool1")
        reg.register(f2, name="tool2")
        schemas = reg.schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "tool1" in names
        assert "tool2" in names

    def test_complex_types_in_schema(self):
        reg = ToolRegistry()
        def process(items: List[str], metadata: Dict[str, str], count: Optional[int] = None) -> str:
            """Process items.

            Args:
                items: List of items.
                metadata: Metadata dict.
                count: Optional count.
            """
            return str(len(items))
        reg.register(process)
        schema = reg.schemas()[0]["function"]["parameters"]
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"
        assert schema["properties"]["metadata"]["type"] == "object"
        assert schema["properties"]["count"]["anyOf"][0]["type"] == "integer"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class TestExecution:
    def test_execute_success(self):
        reg = ToolRegistry()
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b
        reg.register(add)
        assert reg.execute("add", {"a": 2, "b": 3}) == 5

    def test_execute_not_found(self):
        reg = ToolRegistry()
        with pytest.raises(ToolError, match="not found"):
            reg.execute("nonexistent", {})

    def test_execute_safe_success(self):
        reg = ToolRegistry()
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b
        reg.register(add)
        result = reg.execute_safe("add", {"a": 2, "b": 3})
        assert result.success
        assert result.result == 5
        assert result.error is None

    def test_execute_safe_not_found(self):
        reg = ToolRegistry()
        result = reg.execute_safe("nonexistent", {})
        assert not result.success
        assert "not found" in result.error

    def test_execute_safe_runtime_error(self):
        reg = ToolRegistry()
        def divide(a: int, b: int) -> float:
            """Divide."""
            return a / b
        reg.register(divide)
        result = reg.execute_safe("divide", {"a": 10, "b": 0})
        assert not result.success
        assert result.error is not None

    def test_execute_batch(self):
        reg = ToolRegistry()
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b
        def mul(a: int, b: int) -> int:
            """Multiply."""
            return a * b
        reg.register(add)
        reg.register(mul)
        results = reg.execute_batch([
            {"name": "add", "arguments": {"a": 1, "b": 2}},
            {"name": "mul", "arguments": {"a": 3, "b": 4}},
            {"name": "missing", "arguments": {}},
        ])
        assert len(results) == 3
        assert results[0].success and results[0].result == 3
        assert results[1].success and results[1].result == 12
        assert not results[2].success

    def test_execute_with_default_args(self):
        reg = ToolRegistry()
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet.

            Args:
                name: Name.
                greeting: Greeting word.
            """
            return f"{greeting}, {name}!"
        reg.register(greet)
        assert reg.execute("greet", {"name": "Alice"}) == "Hello, Alice!"
        assert reg.execute("greet", {"name": "Bob", "greeting": "Hi"}) == "Hi, Bob!"


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

class TestToolDecorator:
    def test_decorator_registers(self):
        reg = ToolRegistry()

        @tool(reg)
        def search(query: str, limit: int = 10) -> str:
            """Search the web.

            Args:
                query: Search query.
                limit: Max results.
            """
            return f"Results for '{query}' (limit={limit})"

        assert "search" in reg
        assert search.name == "search"

    def test_decorator_custom_name(self):
        reg = ToolRegistry()

        @tool(reg, name="web_search")
        def search(query: str) -> str:
            """Search."""
            return query

        assert "web_search" in reg
        assert "search" not in reg

    def test_decorator_custom_description(self):
        reg = ToolRegistry()

        @tool(reg, description="Custom search tool")
        def search(query: str) -> str:
            """Original docstring."""
            return query

        t = reg.tools["search"]
        assert t.description == "Custom search tool"

    def test_decorator_execution(self):
        reg = ToolRegistry()

        @tool(reg)
        def calculate(expression: str) -> str:
            """Calculate.

            Args:
                expression: Math expression.
            """
            return str(eval(expression))

        result = reg.execute("calculate", {"expression": "2 + 3"})
        assert result == "5"


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_workflow(self):
        """End-to-end: register, generate schemas, execute."""
        reg = ToolRegistry()

        @tool(reg)
        def get_weather(city: str, units: str = "celsius") -> str:
            """Get the current weather for a city.

            Args:
                city: The name of the city.
                units: Temperature units — "celsius" or "fahrenheit".
            """
            return f"Weather in {city}: 22 degrees {units[0].upper()}"

        @tool(reg)
        def get_time(timezone: str = "UTC") -> str:
            """Get the current time.

            Args:
                timezone: Timezone name.
            """
            return f"Time in {timezone}: 12:00"

        # Check registration
        assert len(reg) == 2
        assert "get_weather" in reg
        assert "get_time" in reg

        # Generate schemas for all providers
        for provider in ("openai", "anthropic", "gemini"):
            schemas = reg.schemas(provider)
            assert len(schemas) == 2

        # Execute tools
        weather = reg.execute("get_weather", {"city": "London"})
        assert "London" in weather
        assert "C" in weather

        time_result = reg.execute("get_time", {"timezone": "PST"})
        assert "PST" in time_result

        # Batch execution
        results = reg.execute_batch([
            {"name": "get_weather", "arguments": {"city": "Paris", "units": "fahrenheit"}},
            {"name": "get_time", "arguments": {}},
        ])
        assert all(r.success for r in results)
        assert "Paris" in results[0].result
        assert "UTC" in results[1].result

        # Info
        info = reg.info()
        assert "get_weather" in info
        assert "city" in info["get_weather"]["required"]
        assert "units" not in info["get_weather"]["required"]
