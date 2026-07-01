"""
Advanced data-structure utilities for everyday developer workflows.

This module provides a collection of small, well-tested helper functions and
classes that simplify common operations on dictionaries, iterables, and
general data structures.  Every function is pure (no side-effects on inputs)
unless explicitly documented otherwise, fully type-hinted, and safe for use
in both synchronous scripts and library code.

Typical use-cases
------------------
* Merging configuration dictionaries without losing nested keys.
* Flattening / unflattening nested dicts for form-processing, URL query
  construction, or serialisation to environment variables.
* Chunking large iterables for batch API calls or paginated processing.
* De-duplicating lists while preserving insertion order.
* Grouping records by a computed key (think ``itertools.groupby`` but
  without the requirement to pre-sort).
* Accessing or setting deeply-nested dict values with dot-notation paths.
* Using an immutable, hashable mapping (``FrozenDict``) as a dict key or
  set member.

Examples
--------
>>> from ilovetools.utils.data_structures import (
...     deep_merge, flatten_dict, unflatten_dict, deep_get, deep_set,
...     chunked, deduplicate, invert_dict, group_by, FrozenDict,
... )

>>> deep_merge({"a": 1, "b": {"x": 10}}, {"b": {"y": 20}, "c": 3})
{'a': 1, 'b': {'x': 10, 'y': 20}, 'c': 3}

>>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
{'a.b': 1, 'a.c': 2, 'd': 3}

>>> unflatten_dict({"a.b": 1, "a.c": 2, "d": 3})
{'a': {'b': 1, 'c': 2}, 'd': 3}

>>> deep_get({"a": {"b": {"c": 42}}}, "a.b.c")
42

>>> list(chunked(range(7), 3))
[[0, 1, 2], [3, 4, 5], [6]]

>>> deduplicate([1, 2, 2, 3, 1, 4])
[1, 2, 3, 4]

>>> invert_dict({"a": 1, "b": 2})
{1: 'a', 2: 'b'}

>>> group_by([1, 2, 3, 4, 5, 6], lambda x: x % 2)
{1: [1, 3, 5], 0: [2, 4, 6]}

>>> fd = FrozenDict({"x": 1, "y": 2})
>>> fd["x"]
1
>>> hash(fd)  # doctest: +SKIP
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from typing import Any, Dict, List, Optional, TypeVar, Union

__all__ = [
    "deep_merge",
    "flatten_dict",
    "unflatten_dict",
    "deep_get",
    "deep_set",
    "deep_delete",
    "chunked",
    "deduplicate",
    "invert_dict",
    "group_by",
    "FrozenDict",
]

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


# ---------------------------------------------------------------------------
#  Dictionary helpers
# ---------------------------------------------------------------------------

def deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
    *,
    max_depth: int = 100,
) -> Dict[str, Any]:
    """Recursively merge *override* into *base* and return a new dict.

    Both *base* and *override* are left untouched — the function always
    returns a freshly-allocated dictionary.

    Nested dictionaries are merged key-by-key.  When the same key exists
    in both and both values are dicts, the merge recurses.  Otherwise the
    value from *override* wins.

    Args:
        base:      The base dictionary (lower priority).
        override:  The override dictionary (higher priority).
        max_depth: Safety guard against infinite recursion caused by
                   self-referencing structures.  Defaults to 100.

    Returns:
        A new dictionary containing the merged result.

    Raises:
        RecursionError: If *max_depth* is exceeded.

    Examples:
        >>> deep_merge({"a": 1}, {"b": 2})
        {'a': 1, 'b': 2}

        >>> deep_merge({"a": {"x": 1}}, {"a": {"y": 2}})
        {'a': {'x': 1, 'y': 2}}

        >>> deep_merge({"a": 1}, {"a": 2})
        {'a': 2}

        >>> deep_merge({}, {"a": 1})
        {'a': 1}
    """
    if max_depth <= 0:
        raise RecursionError("max_depth exceeded in deep_merge — possible circular reference")
    result: Dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value, max_depth=max_depth - 1)
        else:
            result[key] = copy.deepcopy(value)
    return result


def flatten_dict(
    d: Mapping[str, Any],
    *,
    separator: str = ".",
    parent_key: str = "",
    max_depth: int = 100,
) -> Dict[str, Any]:
    """Flatten a nested mapping into a single-level dict with compound keys.

    Args:
        d:         The nested mapping to flatten.
        separator: String used to join nested key segments.
        parent_key:  (internal) prefix for recursive calls.
        max_depth: Safety guard against infinite recursion.

    Returns:
        A flat dictionary whose keys are dot-separated paths.

    Raises:
        RecursionError: If *max_depth* is exceeded.

    Examples:
        >>> flatten_dict({"a": 1})
        {'a': 1}

        >>> flatten_dict({"a": {"b": {"c": 1}}})
        {'a.b.c': 1}

        >>> flatten_dict({"a": {"b": 1}}, separator="/")
        {'a/b': 1}

        >>> flatten_dict({})
        {}
    """
    if max_depth <= 0:
        raise RecursionError("max_depth exceeded in flatten_dict — possible circular reference")
    items: Dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            items.update(flatten_dict(value, separator=separator, parent_key=new_key, max_depth=max_depth - 1))
        else:
            items[new_key] = value
    return items


def unflatten_dict(
    d: Mapping[str, Any],
    *,
    separator: str = ".",
) -> Dict[str, Any]:
    """Reverse of :func:`flatten_dict` — expand dot-separated keys into nested dicts.

    Args:
        d:         A flat mapping with compound keys.
        separator: The separator used in the keys.

    Returns:
        A nested dictionary.

    Examples:
        >>> unflatten_dict({"a.b.c": 1})
        {'a': {'b': {'c': 1}}}

        >>> unflatten_dict({"a": 1, "b.c": 2})
        {'a': 1, 'b': {'c': 2}}

        >>> unflatten_dict({"a/b": 1}, separator="/")
        {'a': {'b': 1}}

        >>> unflatten_dict({})
        {}
    """
    result: Dict[str, Any] = {}
    for compound_key, value in d.items():
        parts = str(compound_key).split(separator)
        current = result
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def deep_get(
    d: Mapping[str, Any],
    key: str,
    default: Any = None,
    *,
    separator: str = ".",
) -> Any:
    """Retrieve a value from a nested mapping using a dot-notation path.

    Args:
        d:         The nested mapping to search.
        key:       Dot-separated path (e.g. ``"a.b.c"``).
        default:   Value to return if the path does not exist.
        separator: Path separator (default ``"."``).

    Returns:
        The value at *key* or *default* if any segment is missing.

    Examples:
        >>> deep_get({"a": {"b": {"c": 42}}}, "a.b.c")
        42

        >>> deep_get({"a": {"b": {}}}, "a.b.c", "fallback")
        'fallback'

        >>> deep_get({"a": 1}, "a")
        1

        >>> deep_get({}, "x.y.z", None) is None
        True
    """
    current: Any = d
    for part in str(key).split(separator):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def deep_set(
    d: Dict[str, Any],
    key: str,
    value: Any,
    *,
    separator: str = ".",
) -> Dict[str, Any]:
    """Set a value in a nested dictionary using a dot-notation path.

    Intermediate dictionaries are created automatically when they don't
    exist.  The input *d* is modified **in place** and also returned for
    convenience.

    Args:
        d:         The target dictionary (mutated in place).
        key:       Dot-separated path.
        value:     The value to set.
        separator: Path separator.

    Returns:
        The same dictionary *d* (for chaining).

    Examples:
        >>> d = {}
        >>> deep_set(d, "a.b.c", 42)
        {'a': {'b': {'c': 42}}}

        >>> d = {"a": {"b": 1}}
        >>> deep_set(d, "a.c", 2)
        {'a': {'b': 1, 'c': 2}}

        >>> deep_set({}, "x", 99)
        {'x': 99}
    """
    parts = str(key).split(separator)
    current = d
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return d


def deep_delete(
    d: Dict[str, Any],
    key: str,
    *,
    separator: str = ".",
) -> bool:
    """Delete a key from a nested dictionary using dot-notation.

    Args:
        d:         The target dictionary (mutated in place).
        key:       Dot-separated path to the key to remove.
        separator: Path separator.

    Returns:
        ``True`` if the key was found and deleted, ``False`` otherwise.

    Examples:
        >>> d = {"a": {"b": 1, "c": 2}}
        >>> deep_delete(d, "a.b")
        True
        >>> d
        {'a': {'c': 2}}

        >>> deep_delete(d, "a.x")
        False

        >>> deep_delete({"x": 1}, "x")
        True
    """
    parts = str(key).split(separator)
    current: Any = d
    for part in parts[:-1]:
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    if isinstance(current, dict) and parts[-1] in current:
        del current[parts[-1]]
        return True
    return False


# ---------------------------------------------------------------------------
#  Iterable helpers
# ---------------------------------------------------------------------------

def chunked(
    iterable: Iterable[T],
    size: int,
) -> Iterator[List[T]]:
    """Yield successive *size*-element chunks from *iterable*.

    The final chunk may be shorter than *size* if the iterable does not
    divide evenly.

    Args:
        iterable: Any iterable (list, range, generator, …).
        size:     Chunk size (must be ≥ 1).

    Yields:
        Lists of at most *size* elements.

    Raises:
        ValueError: If *size* is less than 1.

    Examples:
        >>> list(chunked(range(7), 3))
        [[0, 1, 2], [3, 4, 5], [6]]

        >>> list(chunked([1, 2], 5))
        [[1, 2]]

        >>> list(chunked([], 3))
        []
    """
    if size < 1:
        raise ValueError("chunk size must be ≥ 1")
    chunk: List[T] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def deduplicate(
    iterable: Iterable[T],
    *,
    key: Optional[Callable[[T], Hashable]] = None,
) -> List[T]:
    """Remove duplicates from *iterable* while preserving insertion order.

    Args:
        iterable: The input iterable.
        key:      Optional function that maps each item to a hashable
                  identity.  If *None*, the items themselves must be
                  hashable.

    Returns:
        A new list with duplicates removed.

    Examples:
        >>> deduplicate([1, 2, 2, 3, 1, 4])
        [1, 2, 3, 4]

        >>> deduplicate(["b", "a", "b", "c"])
        ['b', 'a', 'c']

        >>> deduplicate([{"id": 1}, {"id": 1}], key=lambda x: x["id"])
        [{'id': 1}]

        >>> deduplicate([])
        []
    """
    seen: set = set()
    result: List[T] = []
    for item in iterable:
        identity = key(item) if key else item
        if identity not in seen:
            seen.add(identity)
            result.append(item)
    return result


def invert_dict(d: Mapping[K, V]) -> Dict[V, K]:
    """Return a new dict with keys and values swapped.

    If multiple keys share the same value, the **last** key wins.

    Args:
        d: The mapping to invert.

    Returns:
        A new dictionary with values as keys and keys as values.

    Raises:
        TypeError: If any value is not hashable.

    Examples:
        >>> invert_dict({"a": 1, "b": 2})
        {1: 'a', 2: 'b'}

        >>> invert_dict({1: "x", 2: "y"})
        {'x': 1, 'y': 2}

        >>> invert_dict({})
        {}
    """
    return {value: key for key, value in d.items()}


def group_by(
    iterable: Iterable[T],
    key_func: Callable[[T], K],
) -> Dict[K, List[T]]:
    """Group items of *iterable* by a computed key.

    Unlike :func:`itertools.groupby`, the input does **not** need to be
    sorted — all items are consumed and grouped in a single pass.

    Args:
        iterable:  The input iterable.
        key_func:  Function that maps each item to a group key.

    Returns:
        A dict mapping each group key to the list of items in that group.

    Examples:
        >>> group_by([1, 2, 3, 4, 5, 6], lambda x: x % 2)
        {1: [1, 3, 5], 0: [2, 4, 6]}

        >>> group_by(["apple", "avocado", "banana"], lambda s: s[0])
        {'a': ['apple', 'avocado'], 'b': ['banana']}

        >>> group_by([], lambda x: x)
        {}
    """
    result: Dict[K, List[T]] = {}
    for item in iterable:
        group_key = key_func(item)
        result.setdefault(group_key, []).append(item)
    return result


# ---------------------------------------------------------------------------
#  Immutable mapping
# ---------------------------------------------------------------------------

class FrozenDict(Mapping):
    """An immutable, hashable dictionary.

    Once created, a ``FrozenDict`` cannot be modified.  It supports the
    full read-only mapping protocol (``len``, ``in``, iteration, ``[]``)
    and is hashable, making it suitable as a dict key or set member.

    Examples:
        >>> fd = FrozenDict({"x": 1, "y": 2})
        >>> fd["x"]
        1

        >>> len(fd)
        2

        >>> "y" in fd
        True

        >>> sorted(fd.keys())
        ['x', 'y']

        >>> # FrozenDict is hashable
        >>> d = {FrozenDict({"a": 1}): "value"}
        >>> d[FrozenDict({"a": 1})]
        'value'
    """

    __slots__ = ("_data", "_hash")

    def __init__(self, data: Optional[Mapping[Any, Any]] = None, /, **kwargs: Any) -> None:
        """Create a FrozenDict from a mapping and/or keyword arguments.

        Args:
            data:  An optional mapping to initialise from.
            **kwargs: Additional key-value pairs.

        Examples:
            >>> FrozenDict({"a": 1})
            FrozenDict({'a': 1})

            >>> FrozenDict(a=1, b=2)
            FrozenDict({'a': 1, 'b': 2})
        """
        merged: Dict[Any, Any] = {}
        if data is not None:
            merged.update(data)
        merged.update(kwargs)
        # Store a tuple of sorted items for deterministic hashing
        self._data: Dict[Any, Any] = dict(merged)
        self._hash: Optional[int] = None

    # -- Mapping protocol ---------------------------------------------------

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    # -- Hashing & equality -------------------------------------------------

    def __hash__(self) -> int:
        if self._hash is None:
            # Hash the sorted items tuple for determinism
            self._hash = hash(tuple(sorted(self._data.items(), key=lambda kv: hash(kv[0]))))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenDict):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    # -- Representation -----------------------------------------------------

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __str__(self) -> str:
        return f"FrozenDict({self._data!r})"

    # -- Explicitly disable mutation ---------------------------------------

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError(f"{type(self).__name__!r} object does not support item assignment")

    def __delitem__(self, key: Any) -> None:
        raise TypeError(f"{type(self).__name__!r} object does not support item deletion")

    def clear(self) -> None:
        raise TypeError(f"{type(self).__name__!r} object is immutable")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"{type(self).__name__!r} object is immutable")

    def popitem(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"{type(self).__name__!r} object is immutable")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(f"{type(self).__name__!r} object is immutable")

    def setdefault(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"{type(self).__name__!r} object is immutable")
