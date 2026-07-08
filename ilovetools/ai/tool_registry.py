"""
Lightweight LLM tool-calling registry with automatic JSON-schema generation.

This module provides a minimal, zero-dependency framework for converting
ordinary Python functions into LLM-callable tools.  It introspects type
hints and docstrings to generate provider-ready JSON schemas (OpenAI /
Anthropic / Gemini function-calling format) and dispatches tool-call
results back to the original function.

Every class and function is pure-Python, fully type-hinted, and depends
only on the standard library (inspect, json, typing, re).

Quick start
-----------
.. code-block:: python

    from ilovetools.ai.tool_registry import ToolRegistry, tool

    registry = ToolRegistry()

    @tool(registry)
    def get_weather(city: str, units: str = "celsius") -> str:
        # Get the current weather for a city.
        # Args: city (str), units (str) -- "celsius" or "fahrenheit"
        return f"Weather in {city}: 22 degrees {units[0].upper()}"

    # Generate schemas for the LLM API
    schemas = registry.schemas()                # OpenAI format
    anthropic_schemas = registry.schemas("anthropic")

    # Execute a tool call from the LLM
    result = registry.execute("get_weather", {"city": "London", "units": "celsius"})
    print(result)  # "Weather in London: 22 degrees C"
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union, get_args, get_origin

__all__ = [
    "ToolRegistry",
    "Tool",
    "tool",
    "ToolError",
    "ToolResult",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ToolError(Exception):
    """Raised when a tool call fails (not found, bad args, execution error)."""


# ---------------------------------------------------------------------------
# Type mapping: Python type -> JSON Schema type string
# ---------------------------------------------------------------------------

_TYPE_MAP: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}

# Provider format names supported by ToolRegistry.schemas
_PROVIDERS = ("openai", "anthropic", "gemini")


def _python_type_to_json_schema(annotation: Any) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON-schema fragment.

    Parameters
    ----------
    annotation : Any
        A Python type annotation (e.g. str, Optional[int], List[str]).

    Returns
    -------
    dict
        A JSON-schema dictionary with type and optional items, anyOf keys.

    Examples
    --------
    >>> _python_type_to_json_schema(str)
    {'type': 'string'}
    >>> _python_type_to_json_schema(int)
    {'type': 'integer'}
    >>> _python_type_to_json_schema(Optional[str])
    {'anyOf': [{'type': 'string'}, {'type': 'null'}]}
    >>> _python_type_to_json_schema(List[int])
    {'type': 'array', 'items': {'type': 'integer'}}
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        has_none = any(a is type(None) for a in args)
        if has_none and len(non_none) == 1:
            inner = _python_type_to_json_schema(non_none[0])
            return {"anyOf": [inner, {"type": "null"}]}
        return {"anyOf": [_python_type_to_json_schema(a) for a in args]}

    if origin in (list, List, Sequence):
        item_schema = _python_type_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    if origin in (dict, Dict):
        val_schema = _python_type_to_json_schema(args[1]) if len(args) >= 2 else {"type": "string"}
        return {"type": "object", "additionalProperties": val_schema}

    if origin in (tuple, Tuple):
        if args and len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        if args:
            return {"type": "array", "items": {"anyOf": [_python_type_to_json_schema(a) for a in args]}}
        return {"type": "array"}

    if origin is not None and hasattr(origin, "__name__") and origin.__name__ == "Literal":
        return {"type": "string", "enum": list(args)}

    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}

    return {"type": "string"}


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------

def _parse_docstring_args(doc: str) -> Dict[str, str]:
    """Extract parameter descriptions from a Google-style docstring.

    Looks for an Args: (or Parameters:) section and parses name: description lines.

    Parameters
    ----------
    doc : str
        The raw docstring text.

    Returns
    -------
    dict[str, str]
        Mapping of parameter name to description string.

    Examples
    --------
    >>> doc = "Summary.\n\nArgs:\n    city: The city name.\n    units: Temp units."
    >>> _parse_docstring_args(doc)
    {'city': 'The city name.', 'units': 'Temp units.'}
    >>> _parse_docstring_args("No args here.")
    {}
    """
    if not doc:
        return {}

    pattern = r"(?:Args|Parameters)\s*:\s*\n(.*?)(?=\n\s*\n|\n\s*(?:Returns|Raises|Yields|Examples|Note|See Also)|$)"
    match = re.search(pattern, doc, re.IGNORECASE | re.DOTALL)
    if not match:
        return {}

    block = match.group(1)
    result: Dict[str, str] = {}
    current_param: Optional[str] = None
    current_desc: List[str] = []

    for line in block.split("\n"):
        param_match = re.match(r"^\s+(\w+)\s*(?:\([^)]+\))?\s*:\s*(.*)", line)
        if param_match:
            if current_param:
                result[current_param] = " ".join(current_desc).strip()
            current_param = param_match.group(1)
            current_desc = [param_match.group(2)] if param_match.group(2) else []
        elif current_param and line.strip():
            current_desc.append(line.strip())
        elif current_param and not line.strip():
            result[current_param] = " ".join(current_desc).strip()
            current_param = None
            current_desc = []

    if current_param:
        result[current_param] = " ".join(current_desc).strip()

    return result


def _parse_docstring_summary(doc: str) -> str:
    """Extract the summary (first paragraph) from a docstring.

    Parameters
    ----------
    doc : str
        The raw docstring text.

    Returns
    -------
    str
        The first paragraph of the docstring, stripped and cleaned.

    Examples
    --------
    >>> _parse_docstring_summary("Get weather.\n\nArgs:\n    city: name")
    'Get weather.'
    >>> _parse_docstring_summary("Single line summary.")
    'Single line summary.'
    >>> _parse_docstring_summary("")
    ''
    """
    if not doc:
        return ""
    lines = doc.strip().split("\n")
    summary_lines: List[str] = []
    for line in lines:
        if not line.strip() and summary_lines:
            break
        if line.strip():
            summary_lines.append(line.strip())
    return " ".join(summary_lines)


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class Tool:
    """Represents a single LLM-callable tool.

    A Tool wraps a Python function with its auto-generated JSON schema,
    making it easy to pass to LLM APIs and execute on demand.

    Attributes
    ----------
    name : str
        The tool name (function name or custom name).
    func : Callable
        The underlying Python function.
    schema : dict
        The JSON-schema describing the function parameters.
    """

    def __init__(self, name: str, func: Callable, description: Optional[str] = None) -> None:
        """Initialise a Tool from a name and callable.

        Parameters
        ----------
        name : str
            The tool name shown to the LLM.
        func : Callable
            The Python function to wrap.
        description : str, optional
            Override the docstring summary as the tool description.
        """
        self.name = name
        self.func = func
        self._sig = inspect.signature(func)
        doc = func.__doc__ or ""
        self.description = description or _parse_docstring_summary(doc)
        self._arg_docs = _parse_docstring_args(doc)
        self.schema = self._build_schema()

    def _build_schema(self) -> Dict[str, Any]:
        """Build the JSON-schema for this tool parameters."""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in self._sig.parameters.items():
            if param_name == "self":
                continue
            prop = _python_type_to_json_schema(param.annotation)
            desc = self._arg_docs.get(param_name, "")
            if desc:
                prop["description"] = desc
            properties[param_name] = prop

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def execute(self, arguments: Dict[str, Any]) -> Any:
        """Execute the wrapped function with the given arguments.

        Parameters
        ----------
        arguments : dict
            Keyword arguments to pass to the function.

        Returns
        -------
        Any
            The return value of the wrapped function.

        Raises
        ------
        ToolError
            If the function raises an exception or arguments are invalid.
        """
        try:
            return self.func(**arguments)
        except TypeError as exc:
            raise ToolError(f"Invalid arguments for tool '{self.name}': {exc}") from exc
        except Exception as exc:
            raise ToolError(f"Tool '{self.name}' execution failed: {exc}") from exc

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r}, params={list(self.schema['properties'].keys())})"


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

class ToolResult:
    """Represents the result of a tool execution.

    Attributes
    ----------
    name : str
        The tool name that was executed.
    success : bool
        Whether the execution succeeded.
    result : Any
        The return value (on success) or error message (on failure).
    error : str or None
        Error message if the execution failed.
    """

    def __init__(self, name: str, success: bool, result: Any, error: Optional[str]) -> None:
        self.name = name
        self.success = success
        self.result = result
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary suitable for JSON encoding."""
        return {
            "name": self.name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return f"ToolResult({status} {self.name}: {self.result if self.success else self.error})"


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """A registry of LLM-callable tools with schema generation and execution.

    The ToolRegistry is the central object: you register Python functions
    (either via the @tool decorator or register() method), generate
    provider-specific JSON schemas for LLM API calls, and execute tool-call
    requests returned by the LLM.

    Attributes
    ----------
    tools : dict[str, Tool]
        Mapping of tool name to Tool instance.
    """

    def __init__(self) -> None:
        """Create an empty tool registry."""
        self.tools: Dict[str, Tool] = {}

    def register(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Tool:
        """Register a Python function as an LLM-callable tool.

        Parameters
        ----------
        func : Callable
            The function to register.
        name : str, optional
            Custom tool name (defaults to func.__name__).
        description : str, optional
            Custom description (defaults to the docstring summary).

        Returns
        -------
        Tool
            The created Tool instance.
        """
        tool_name = name or func.__name__
        t = Tool(tool_name, func, description)
        self.tools[tool_name] = t
        return t

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Parameters
        ----------
        name : str
            The tool name to remove.

        Raises
        ------
        ToolError
            If the tool is not found.
        """
        if name not in self.tools:
            raise ToolError(f"Tool '{name}' not found in registry")
        del self.tools[name]

    def schemas(self, provider: str = "openai") -> List[Dict[str, Any]]:
        """Generate provider-specific tool schemas for LLM API calls.

        Parameters
        ----------
        provider : str
            One of "openai", "anthropic", or "gemini".
            Defaults to "openai".

        Returns
        -------
        list[dict]
            A list of schema dicts in the provider format.

        Raises
        ------
        ValueError
            If the provider is not supported.
        """
        if provider not in _PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider}'. Supported: {', '.join(_PROVIDERS)}"
            )

        result: List[Dict[str, Any]] = []
        for t in self.tools.values():
            if provider == "openai":
                result.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.schema,
                    },
                })
            elif provider == "anthropic":
                result.append({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.schema,
                })
            elif provider == "gemini":
                result.append({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema,
                })
        return result

    def schemas_json(self, provider: str = "openai", indent: int = 2) -> str:
        """Generate provider-specific tool schemas as a JSON string.

        Parameters
        ----------
        provider : str
            One of "openai", "anthropic", or "gemini".
        indent : int
            JSON indentation level.

        Returns
        -------
        str
            JSON-encoded schema array.
        """
        return json.dumps(self.schemas(provider), indent=indent)

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered tool by name.

        Parameters
        ----------
        name : str
            The tool name.
        arguments : dict
            Keyword arguments for the function.

        Returns
        -------
        Any
            The function return value.

        Raises
        ------
        ToolError
            If the tool is not found or execution fails.
        """
        if name not in self.tools:
            raise ToolError(f"Tool '{name}' not found in registry")
        return self.tools[name].execute(arguments)

    def execute_safe(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool and return a ToolResult (never raises).

        Unlike execute(), this method catches all exceptions and returns
        them as part of the ToolResult instead of raising.

        Parameters
        ----------
        name : str
            The tool name.
        arguments : dict
            Keyword arguments for the function.

        Returns
        -------
        ToolResult
            A result object with success, result, and error.
        """
        try:
            result = self.execute(name, arguments)
            return ToolResult(name, True, result, None)
        except ToolError as exc:
            return ToolResult(name, False, None, str(exc))

    def execute_batch(self, calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute multiple tool calls in sequence.

        Parameters
        ----------
        calls : list[dict]
            Each dict must have "name" (str) and "arguments" (dict).

        Returns
        -------
        list[ToolResult]
            One ToolResult per call, in order.
        """
        results: List[ToolResult] = []
        for call in calls:
            name = call.get("name", "")
            arguments = call.get("arguments", {})
            results.append(self.execute_safe(name, arguments))
        return results

    def list_names(self) -> List[str]:
        """Return a sorted list of registered tool names.

        Returns
        -------
        list[str]
            Alphabetically sorted tool names.
        """
        return sorted(self.tools.keys())

    def info(self) -> Dict[str, Dict[str, Any]]:
        """Return detailed information about all registered tools.

        Returns
        -------
        dict[str, dict]
            Mapping of tool name to a dict with description, parameters, required.
        """
        return {
            name: {
                "description": t.description,
                "parameters": list(t.schema["properties"].keys()),
                "required": t.schema["required"],
            }
            for name, t in self.tools.items()
        }

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self)}, names={self.list_names()})"


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def tool(
    registry: ToolRegistry,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable], Tool]:
    """Decorator that registers a function as an LLM-callable tool.

    Parameters
    ----------
    registry : ToolRegistry
        The registry to register the tool in.
    name : str, optional
        Custom tool name (defaults to func.__name__).
    description : str, optional
        Custom description (defaults to the docstring summary).

    Returns
    -------
    Callable
        A decorator that registers the function and returns the Tool instance.
    """
    def decorator(func: Callable) -> Tool:
        return registry.register(func, name=name, description=description)
    return decorator
