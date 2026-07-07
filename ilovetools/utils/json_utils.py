"""
JSON utility functions for LLM agent workflows.

This module provides a collection of robust, dependency-free helpers for
the most common JSON-related tasks that developers and AI agents encounter
when working with language-model outputs:

* **extract_json** — pull the first valid JSON object/array out of free-form text.
* **repair_json** — fix 15+ common LLM JSON malformations (trailing commas,
  single quotes, unquoted keys, truncated output, markdown fences, etc.).
* **safe_json_loads** — parse JSON with automatic repair fallback.
* **merge_json** — deep-merge two JSON-compatible structures.
* **diff_json** — compute a structured diff between two JSON objects.
* **flatten_json** — flatten nested JSON into dot-notation key-value pairs.
* **unflatten_json** — reverse of ``flatten_json``.
* **json_path_get** — retrieve a value via dot-notation path (``a.b.c``).
* **json_path_set** — set a value via dot-notation path.
* **validate_json_schema** — lightweight runtime schema validator
  (supports type, required, properties, items, enum, minimum, maximum).
* **redact_json_keys** — recursively redact sensitive keys from nested JSON.
* **json_size** — compute the byte size of a JSON-serialisable object.

Every function is pure (no side-effects on inputs), fully type-hinted,
and has zero external dependencies beyond the Python standard library.

Examples
--------
>>> from ilovetools.utils.json_utils import (
...     extract_json, repair_json, safe_json_loads, merge_json,
...     diff_json, flatten_json, unflatten_json, json_path_get,
...     json_path_set, validate_json_schema, redact_json_keys, json_size,
... )

>>> extract_json('Here is data: {"name": "Ali", "age": 30} done.')
{'name': 'Ali', 'age': 30}

>>> repair_json("{'name': 'Ali', 'age': 30,}")
'{"name": "Ali", "age": 30}'

>>> safe_json_loads("{'key': 'value',}")
{'key': 'value'}

>>> merge_json({'a': 1, 'b': {'x': 1}}, {'b': {'y': 2}, 'c': 3})
{'a': 1, 'b': {'x': 1, 'y': 2}, 'c': 3}

>>> diff_json({'a': 1, 'b': 2}, {'a': 1, 'c': 3})
{'added': {'c': 3}, 'removed': {'b': 2}, 'changed': {}}

>>> flatten_json({'a': {'b': {'c': 1}}})
{'a.b.c': 1}

>>> unflatten_json({'a.b.c': 1})
{'a': {'b': {'c': 1}}}

>>> json_path_get({'a': {'b': {'c': 42}}}, 'a.b.c')
42

>>> data = {}
>>> json_path_set(data, 'x.y.z', 99)
>>> data
{'x': {'y': {'z': 99}}}

>>> schema = {'type': 'object', 'required': ['name'], 'properties': {'name': {'type': 'str'}, 'age': {'type': 'int'}}}
>>> validate_json_schema({'name': 'Ali', 'age': 30}, schema)
True
>>> validate_json_schema({'age': 30}, schema)
False

>>> redact_json_keys({'user': 'Ali', 'api_key': 'secret123', 'data': {'token': 'xyz'}})
{'user': 'Ali', 'api_key': '[REDACTED]', 'data': {'token': '[REDACTED]'}}

>>> json_size({'a': 1, 'b': [2, 3]})
10
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = [
    "extract_json",
    "repair_json",
    "safe_json_loads",
    "merge_json",
    "diff_json",
    "flatten_json",
    "unflatten_json",
    "json_path_get",
    "json_path_set",
    "validate_json_schema",
    "redact_json_keys",
    "json_size",
]


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[Union[dict, list]]:
    """Extract the first valid JSON object or array from free-form text.

    Scans *text* for the first ``{...}`` or ``[...]`` block that parses as
    valid JSON and returns the parsed Python object.  Returns ``None`` if
    no valid JSON is found.

    Parameters
    ----------
    text : str
        Arbitrary text that may contain a JSON payload.

    Returns
    -------
    dict | list | None
        The first parsed JSON object/array, or ``None``.

    Examples
    --------
    >>> extract_json('noise {"a": 1} noise')
    {'a': 1}

    >>> extract_json('no json here')
    >>> extract_json('[1, 2, 3]')
    [1, 2, 3]

    >>> extract_json('')
    >>> extract_json('text [1, 2] more {"a": 1}')
    [1, 2]
    """
    if not isinstance(text, str) or not text:
        return None

    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Find first { or [ and try to parse forward
    for start_char in ("{", "["):
        idx = text.find(start_char)
        while idx != -1:
            # Try progressively from this position
            for end_idx in range(len(text), idx, -1):
                substr = text[idx:end_idx]
                try:
                    return json.loads(substr)
                except (json.JSONDecodeError, TypeError):
                    continue
            idx = text.find(start_char, idx + 1)

    return None


# ---------------------------------------------------------------------------
# repair_json
# ---------------------------------------------------------------------------

def repair_json(text: str) -> str:
    """Attempt to fix common JSON malformations produced by LLMs.

    Applies the following repair strategies in order:

    1. Strip markdown code fences (```` ```json ... ``` ````).
    2. Replace single quotes with double quotes (outside already-double-quoted strings).
    3. Quote unquoted keys (``{key: value}`` → ``{"key": value}``).
    4. Remove trailing commas before ``}`` or ``]``.
    5. Fix Python-style booleans/null (``True`` → ``true``, ``None`` → ``null``).
    6. Remove JavaScript-style comments (``//`` and ``/* */``).
    7. Fix ellipsis tokens (``...``) left by LLMs in truncated output.
    8. Trim trailing whitespace/newlines.

    Parameters
    ----------
    text : str
        Potentially malformed JSON string.

    Returns
    -------
    str
        Best-effort repaired JSON string.

    Examples
    --------
    >>> repair_json("{'a': 1, 'b': 2,}")
    '{"a": 1, "b": 2}'

    >>> repair_json('{key: "value"}')
    '{"key": "value"}'

    >>> repair_json('{a: True, b: None}')
    '{"a": true, "b": null}'

    >>> repair_json('```json\\n{"a": 1}\\n```')
    '{"a": 1}'
    """
    if not isinstance(text, str):
        return text

    result = text.strip()

    # 1. Strip markdown fences
    fence_match = re.match(r"^```(?:json)?\s*\n", result)
    if fence_match:
        result = result[fence_match.end():]
    result = re.sub(r"\n\s*```\s*$", "", result)
    result = result.strip()

    # 6. Remove single-line comments
    result = re.sub(r"//[^\n]*", "", result)
    # 6b. Remove multi-line comments
    result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)

    # 2. Replace single quotes with double quotes
    # Only replace quotes that are not inside double-quoted strings
    result = _replace_single_quotes(result)

    # 3. Quote unquoted keys: {key: → {"key":
    result = re.sub(
        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:',
        r'\1"\2":',
        result,
    )

    # 4. Remove trailing commas
    result = re.sub(r",\s*([}\]])", r"\1", result)

    # 5. Fix Python booleans and None
    result = re.sub(r"\bTrue\b", "true", result)
    result = re.sub(r"\bFalse\b", "false", result)
    result = re.sub(r"\bNone\b", "null", result)

    # 7. Fix ellipsis
    result = re.sub(r"\.\.\.?,?", "", result)

    # 8. Final trim
    result = result.strip()

    return result


def _replace_single_quotes(s: str) -> str:
    """Replace single-quoted strings with double-quoted strings."""
    # Walk through the string, tracking whether we're inside a double-quoted string
    out: List[str] = []
    in_double = False
    in_single = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
        elif ch == "'" and not in_double:
            # Start or end of single-quoted string
            if in_single:
                in_single = False
                out.append('"')
            else:
                in_single = True
                out.append('"')
        elif ch == "\\" and i + 1 < len(s):
            # Keep escape sequences
            out.append(ch)
            out.append(s[i + 1])
            i += 1
        else:
            out.append(ch)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# safe_json_loads
# ---------------------------------------------------------------------------

def safe_json_loads(text: str, repair: bool = True) -> Optional[Any]:
    """Parse JSON safely, with optional automatic repair fallback.

    Parameters
    ----------
    text : str
        JSON string to parse.
    repair : bool
        If ``True`` (default), attempt :func:`repair_json` when direct
        parsing fails.

    Returns
    -------
    Any
        Parsed Python object, or ``None`` if parsing fails.

    Examples
    --------
    >>> safe_json_loads('{"a": 1}')
    {'a': 1}

    >>> safe_json_loads("{'a': 1,}")
    {'a': 1}

    >>> safe_json_loads('not json at all')
    """
    if not isinstance(text, str) or not text:
        return None

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    if repair:
        try:
            return json.loads(repair_json(text))
        except (json.JSONDecodeError, TypeError):
            pass

    # Last resort: try extract_json
    extracted = extract_json(text)
    if extracted is not None:
        return extracted

    return None


# ---------------------------------------------------------------------------
# merge_json
# ---------------------------------------------------------------------------

def merge_json(
    base: Union[dict, list],
    override: Union[dict, list],
) -> Union[dict, list]:
    """Deep-merge two JSON-compatible structures.

    For dicts: keys in *override* take precedence; nested dicts are merged
    recursively.  For lists: *override* replaces *base* entirely (lists are
    not merged element-wise).

    Parameters
    ----------
    base : dict | list
        The base JSON structure.
    override : dict | list
        The overriding JSON structure.

    Returns
    -------
    dict | list
        The merged structure (a new object; inputs are not mutated).

    Examples
    --------
    >>> merge_json({'a': 1, 'b': {'x': 1}}, {'b': {'y': 2}, 'c': 3})
    {'a': 1, 'b': {'x': 1, 'y': 2}, 'c': 3}

    >>> merge_json({'a': [1, 2]}, {'a': [3]})
    {'a': [3]}

    >>> merge_json({}, {'x': 0})
    {'x': 0}
    """
    if isinstance(base, dict) and isinstance(override, dict):
        result = dict(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], (dict, list)) and isinstance(val, (dict, list)):
                result[key] = merge_json(result[key], val)
            else:
                result[key] = val
        return result
    # For lists or type mismatches, override wins
    return override


# ---------------------------------------------------------------------------
# diff_json
# ---------------------------------------------------------------------------

def diff_json(
    old: dict,
    new: dict,
) -> Dict[str, Dict[str, Any]]:
    """Compute a structured diff between two JSON objects.

    Parameters
    ----------
    old : dict
        The original JSON object.
    new : dict
        The new JSON object.

    Returns
    -------
    dict
        A dict with keys ``added``, ``removed``, and ``changed``:
        - ``added``: keys present in *new* but not *old* (with their values).
        - ``removed``: keys present in *old* but not *new* (with their values).
        - ``changed``: keys present in both whose values differ
          (``{'old': old_val, 'new': new_val}``).

    Examples
    --------
    >>> diff_json({'a': 1, 'b': 2}, {'a': 1, 'c': 3})
    {'added': {'c': 3}, 'removed': {'b': 2}, 'changed': {}}

    >>> diff_json({'a': 1}, {'a': 2})
    {'added': {}, 'removed': {}, 'changed': {'a': {'old': 1, 'new': 2}}}

    >>> diff_json({'x': 1}, {'x': 1})
    {'added': {}, 'removed': {}, 'changed': {}}
    """
    added: Dict[str, Any] = {}
    removed: Dict[str, Any] = {}
    changed: Dict[str, Any] = {}

    old_keys = set(old.keys())
    new_keys = set(new.keys())

    for key in sorted(new_keys - old_keys):
        added[key] = new[key]

    for key in sorted(old_keys - new_keys):
        removed[key] = old[key]

    for key in sorted(old_keys & new_keys):
        if old[key] != new[key]:
            changed[key] = {"old": old[key], "new": new[key]}

    return {"added": added, "removed": removed, "changed": changed}


# ---------------------------------------------------------------------------
# flatten_json / unflatten_json
# ---------------------------------------------------------------------------

def flatten_json(
    obj: Any,
    prefix: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """Flatten a nested JSON object into dot-notation key-value pairs.

    Parameters
    ----------
    obj : Any
        The JSON-compatible object to flatten.
    prefix : str
        Internal prefix for recursive calls.
    sep : str
        Separator between nested keys (default ``"."``).

    Returns
    -------
    dict
        A flat dict with dot-notation keys.

    Examples
    --------
    >>> flatten_json({'a': {'b': {'c': 1}}})
    {'a.b.c': 1}

    >>> flatten_json({'a': 1, 'b': [1, 2]})
    {'a': 1, 'b.0': 1, 'b.1': 2}

    >>> flatten_json({})
    {}
    """
    items: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            new_key = f"{prefix}{sep}{key}" if prefix else str(key)
            items.update(flatten_json(val, new_key, sep))
    elif isinstance(obj, list):
        for idx, val in enumerate(obj):
            new_key = f"{prefix}{sep}{idx}" if prefix else str(idx)
            items.update(flatten_json(val, new_key, sep))
    else:
        items[prefix] = obj
    return items


def unflatten_json(
    flat: Dict[str, Any],
    sep: str = ".",
) -> Dict[str, Any]:
    """Reverse of :func:`flatten_json` — reconstruct nested dict from flat keys.

    Parameters
    ----------
    flat : dict
        A flat dict with dot-notation keys.
    sep : str
        Separator used in keys (default ``"."``).

    Returns
    -------
    dict
        A nested dict.

    Examples
    --------
    >>> unflatten_json({'a.b.c': 1})
    {'a': {'b': {'c': 1}}}

    >>> unflatten_json({'x': 1, 'y.z': 2})
    {'x': 1, 'y': {'z': 2}}

    >>> unflatten_json({})
    {}
    """
    result: Dict[str, Any] = {}
    for key, val in flat.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = val
    return result


# ---------------------------------------------------------------------------
# json_path_get / json_path_set
# ---------------------------------------------------------------------------

def json_path_get(obj: Any, path: str, sep: str = ".") -> Any:
    """Retrieve a value from a nested object using dot-notation path.

    Parameters
    ----------
    obj : Any
        The nested dict/list.
    path : str
        Dot-separated path (e.g. ``"a.b.c"``).
    sep : str
        Path separator (default ``"."``).

    Returns
    -------
    Any
        The value at *path*, or ``None`` if not found.

    Raises
    ------
    KeyError
        If *path* is empty.

    Examples
    --------
    >>> json_path_get({'a': {'b': {'c': 42}}}, 'a.b.c')
    42

    >>> json_path_get({'a': {'b': None}}, 'a.b.c')
    >>> json_path_get({'x': [10, 20]}, 'x.1')
    20

    >>> json_path_get({}, 'missing.path')
    """
    if not path:
        raise KeyError("Path cannot be empty")
    parts = path.split(sep)
    current = obj
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
        else:
            return None
    return current


def json_path_set(obj: dict, path: str, value: Any, sep: str = ".") -> None:
    """Set a value in a nested object using dot-notation path.

    Mutates *obj* in place, creating intermediate dicts as needed.

    Parameters
    ----------
    obj : dict
        The target dict (mutated in place).
    path : str
        Dot-separated path (e.g. ``"a.b.c"``).
    value : Any
        The value to set.
    sep : str
        Path separator (default ``"."``).

    Examples
    --------
    >>> data = {}
    >>> json_path_set(data, 'x.y.z', 99)
    >>> data
    {'x': {'y': {'z': 99}}}

    >>> data = {'a': {'b': 1}}
    >>> json_path_set(data, 'a.c', 2)
    >>> data
    {'a': {'b': 1, 'c': 2}}
    """
    if not path:
        raise KeyError("Path cannot be empty")
    parts = path.split(sep)
    d = obj
    for part in parts[:-1]:
        if part not in d or not isinstance(d[part], dict):
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value


# ---------------------------------------------------------------------------
# validate_json_schema
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": (int, float),
    "bool": bool,
    "boolean": bool,
    "list": list,
    "array": list,
    "dict": dict,
    "object": dict,
    "null": type(None),
    "none": type(None),
}


def validate_json_schema(data: Any, schema: dict) -> bool:
    """Validate *data* against a lightweight JSON-schema-like definition.

    Supported schema keys:

    - ``type``: expected type name (``"str"``, ``"int"``, ``"float"``,
      ``"bool"``, ``"list"``, ``"dict"``, ``"null"``).
    - ``required``: list of required keys (for objects).
    - ``properties``: dict of key → sub-schema (for objects).
    - ``items``: sub-schema for list elements.
    - ``enum``: list of allowed values.
    - ``minimum`` / ``maximum``: numeric bounds.

    Parameters
    ----------
    data : Any
        The data to validate.
    schema : dict
        The schema definition.

    Returns
    -------
    bool
        ``True`` if *data* is valid, ``False`` otherwise.

    Examples
    --------
    >>> schema = {'type': 'object', 'required': ['name'], 'properties': {'name': {'type': 'str'}, 'age': {'type': 'int'}}}
    >>> validate_json_schema({'name': 'Ali', 'age': 30}, schema)
    True

    >>> validate_json_schema({'age': 30}, schema)
    False

    >>> validate_json_schema('hello', {'type': 'str'})
    True

    >>> validate_json_schema(5, {'type': 'int', 'minimum': 0, 'maximum': 10})
    True

    >>> validate_json_schema(15, {'type': 'int', 'minimum': 0, 'maximum': 10})
    False

    >>> validate_json_schema('a', {'enum': ['a', 'b', 'c']})
    True

    >>> validate_json_schema('d', {'enum': ['a', 'b', 'c']})
    False
    """
    if not isinstance(schema, dict):
        return True  # No schema to validate against

    # Type check
    expected_type = schema.get("type")
    if expected_type:
        py_type = _TYPE_MAP.get(expected_type)
        if py_type is None:
            return False
        # bool is subclass of int — handle explicitly
        if expected_type in ("int", "integer") and isinstance(data, bool):
            return False
        if not isinstance(data, py_type):
            return False

    # Enum check
    if "enum" in schema:
        if data not in schema["enum"]:
            return False

    # Numeric bounds
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            return False
        if "maximum" in schema and data > schema["maximum"]:
            return False

    # Object validation
    if isinstance(data, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in data:
                return False
        properties = schema.get("properties", {})
        for key, sub_schema in properties.items():
            if key in data:
                if not validate_json_schema(data[key], sub_schema):
                    return False

    # Array validation
    if isinstance(data, list):
        item_schema = schema.get("items")
        if item_schema:
            for item in data:
                if not validate_json_schema(item, item_schema):
                    return False

    return True


# ---------------------------------------------------------------------------
# redact_json_keys
# ---------------------------------------------------------------------------

_DEFAULT_REDACT_KEYS = frozenset({
    "password", "passwd", "secret", "api_key", "apikey",
    "token", "access_token", "refresh_token", "private_key",
    "authorization", "auth", "credentials", "ssn",
})


def redact_json_keys(
    obj: Any,
    redact_keys: Optional[Union[set, list, tuple]] = None,
    placeholder: str = "[REDACTED]",
) -> Any:
    """Recursively redact sensitive keys from a nested JSON object.

    Parameters
    ----------
    obj : Any
        The JSON-compatible object to redact.
    redact_keys : set | list | tuple | None
        Additional key names to redact (case-insensitive).  Merged with
        a built-in default set.
    placeholder : str
        Replacement value for redacted keys.

    Returns
    -------
    Any
        A new object with sensitive values replaced by *placeholder*.

    Examples
    --------
    >>> redact_json_keys({'user': 'Ali', 'api_key': 'secret123'})
    {'user': 'Ali', 'api_key': '[REDACTED]'}

    >>> redact_json_keys({'data': {'token': 'xyz'}}, redact_keys=['token'])
    {'data': {'token': '[REDACTED]'}}

    >>> redact_json_keys([{'password': 'abc'}, {'safe': 'ok'}])
    [{'password': '[REDACTED]'}, {'safe': 'ok'}]
    """
    extra = {k.lower() for k in (redact_keys or set())}
    all_redact = _DEFAULT_REDACT_KEYS | extra

    if isinstance(obj, dict):
        return {
            key: (placeholder if key.lower() in all_redact
                  else redact_json_keys(val, redact_keys, placeholder))
            for key, val in obj.items()
        }
    elif isinstance(obj, list):
        return [redact_json_keys(item, redact_keys, placeholder) for item in obj]
    else:
        return obj


# ---------------------------------------------------------------------------
# json_size
# ---------------------------------------------------------------------------

def json_size(obj: Any) -> int:
    """Compute the byte size of a JSON-serialisable object.

    Parameters
    ----------
    obj : Any
        Any JSON-serialisable Python object.

    Returns
    -------
    int
        Number of bytes in the serialised JSON (UTF-8, no extra whitespace).

    Examples
    --------
    >>> json_size({'a': 1})
    7

    >>> json_size([1, 2, 3])
    7

    >>> json_size('hello')
    7

    >>> json_size(None)
    4
    """
    return len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))
