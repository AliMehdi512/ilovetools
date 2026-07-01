"""
Comprehensive pytest suite for ilovetools.utils.data_structures

Covers all public functions and the FrozenDict class, including edge
cases, exceptions, immutability guarantees, and typical usage paths.
"""

import pytest
from ilovetools.utils.data_structures import (
    deep_merge,
    flatten_dict,
    unflatten_dict,
    deep_get,
    deep_set,
    deep_delete,
    chunked,
    deduplicate,
    invert_dict,
    group_by,
    FrozenDict,
)


# ===========================================================================
#  deep_merge
# ===========================================================================

class TestDeepMerge:
    def test_simple_merge(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override_wins(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self):
        result = deep_merge({"a": {"x": 1}}, {"a": {"y": 2}})
        assert result == {"a": {"x": 1, "y": 2}}

    def test_nested_override(self):
        result = deep_merge({"a": {"x": 1}}, {"a": {"x": 99}})
        assert result == {"a": {"x": 99}}

    def test_deeply_nested(self):
        base = {"a": {"b": {"c": {"d": 1}}}}
        override = {"a": {"b": {"c": {"e": 2}}}}
        assert deep_merge(base, override) == {"a": {"b": {"c": {"d": 1, "e": 2}}}}

    def test_originals_untouched(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        deep_merge(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}

    def test_empty_base(self):
        assert deep_merge({}, {"a": 1}) == {"a": 1}

    def test_empty_override(self):
        assert deep_merge({"a": 1}, {}) == {"a": 1}

    def test_both_empty(self):
        assert deep_merge({}, {}) == {}

    def test_override_non_dict_replaces_dict(self):
        result = deep_merge({"a": {"x": 1}}, {"a": 42})
        assert result == {"a": 42}

    def test_override_dict_replaces_non_dict(self):
        result = deep_merge({"a": 42}, {"a": {"x": 1}})
        assert result == {"a": {"x": 1}}

    def test_list_values_replaced(self):
        result = deep_merge({"a": [1, 2]}, {"a": [3, 4]})
        assert result == {"a": [3, 4]}

    def test_returns_new_dict(self):
        base = {"a": 1}
        result = deep_merge(base, {})
        assert result is not base


# ===========================================================================
#  flatten_dict
# ===========================================================================

class TestFlattenDict:
    def test_flat_dict(self):
        assert flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        assert flatten_dict({"a": {"b": 1}}) == {"a.b": 1}

    def test_deeply_nested(self):
        assert flatten_dict({"a": {"b": {"c": 1}}}) == {"a.b.c": 1}

    def test_mixed(self):
        assert flatten_dict({"a": {"b": 1, "c": 2}, "d": 3}) == {"a.b": 1, "a.c": 2, "d": 3}

    def test_custom_separator(self):
        assert flatten_dict({"a": {"b": 1}}, separator="/") == {"a/b": 1}

    def test_empty_dict(self):
        assert flatten_dict({}) == {}

    def test_non_string_keys(self):
        assert flatten_dict({1: {2: "val"}}) == {"1.2": "val"}

    def test_values_can_be_anything(self):
        result = flatten_dict({"a": {"b": [1, 2], "c": None}})
        assert result == {"a.b": [1, 2], "a.c": None}


# ===========================================================================
#  unflatten_dict
# ===========================================================================

class TestUnflattenDict:
    def test_simple(self):
        assert unflatten_dict({"a.b.c": 1}) == {"a": {"b": {"c": 1}}}

    def test_mixed(self):
        assert unflatten_dict({"a": 1, "b.c": 2}) == {"a": 1, "b": {"c": 2}}

    def test_custom_separator(self):
        assert unflatten_dict({"a/b": 1}, separator="/") == {"a": {"b": 1}}

    def test_empty(self):
        assert unflatten_dict({}) == {}

    def test_roundtrip(self):
        original = {"a": {"b": {"c": 1}, "d": 2}, "e": 3}
        flat = flatten_dict(original)
        assert unflatten_dict(flat) == original

    def test_multiple_nested(self):
        result = unflatten_dict({"x.y": 1, "x.z": 2, "w": 3})
        assert result == {"x": {"y": 1, "z": 2}, "w": 3}


# ===========================================================================
#  deep_get
# ===========================================================================

class TestDeepGet:
    def test_simple_path(self):
        assert deep_get({"a": 1}, "a") == 1

    def test_nested_path(self):
        assert deep_get({"a": {"b": {"c": 42}}}, "a.b.c") == 42

    def test_missing_key_returns_default(self):
        assert deep_get({"a": 1}, "b", "fallback") == "fallback"

    def test_missing_nested_key_returns_default(self):
        assert deep_get({"a": {"b": {}}}, "a.b.c", "fallback") == "fallback"

    def test_default_none(self):
        assert deep_get({}, "x.y.z") is None

    def test_custom_separator(self):
        assert deep_get({"a": {"b": 1}}, "a/b", separator="/") == 1

    def test_non_dict_intermediate(self):
        assert deep_get({"a": [1, 2]}, "a.b", "default") == "default"

    def test_empty_key(self):
        assert deep_get({"a": 1}, "", "default") == "default"


# ===========================================================================
#  deep_set
# ===========================================================================

class TestDeepSet:
    def test_simple_set(self):
        d = {}
        assert deep_set(d, "a", 1) == {"a": 1}

    def test_nested_set(self):
        d = {}
        assert deep_set(d, "a.b.c", 42) == {"a": {"b": {"c": 42}}}

    def test_overwrite_existing(self):
        d = {"a": {"b": 1}}
        assert deep_set(d, "a.b", 99) == {"a": {"b": 99}}

    def test_add_to_existing(self):
        d = {"a": {"b": 1}}
        assert deep_set(d, "a.c", 2) == {"a": {"b": 1, "c": 2}}

    def test_custom_separator(self):
        d = {}
        deep_set(d, "a/b", 1, separator="/")
        assert d == {"a": {"b": 1}}

    def test_mutates_in_place(self):
        d = {}
        result = deep_set(d, "x", 1)
        assert result is d
        assert d == {"x": 1}

    def test_replaces_non_dict_intermediate(self):
        d = {"a": 1}
        deep_set(d, "a.b", 2)
        assert d == {"a": {"b": 2}}


# ===========================================================================
#  deep_delete
# ===========================================================================

class TestDeepDelete:
    def test_simple_delete(self):
        d = {"a": 1, "b": 2}
        assert deep_delete(d, "a") is True
        assert d == {"b": 2}

    def test_nested_delete(self):
        d = {"a": {"b": 1, "c": 2}}
        assert deep_delete(d, "a.b") is True
        assert d == {"a": {"c": 2}}

    def test_missing_key(self):
        d = {"a": 1}
        assert deep_delete(d, "b") is False
        assert d == {"a": 1}

    def test_missing_nested_key(self):
        d = {"a": {"b": 1}}
        assert deep_delete(d, "a.x") is False

    def test_missing_intermediate(self):
        d = {"a": 1}
        assert deep_delete(d, "x.y.z") is False

    def test_custom_separator(self):
        d = {"a": {"b": 1}}
        assert deep_delete(d, "a/b", separator="/") is True
        assert d == {"a": {}}


# ===========================================================================
#  chunked
# ===========================================================================

class TestChunked:
    def test_even_split(self):
        assert list(chunked(range(6), 3)) == [[0, 1, 2], [3, 4, 5]]

    def test_uneven_split(self):
        assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]

    def test_chunk_larger_than_iterable(self):
        assert list(chunked([1, 2], 5)) == [[1, 2]]

    def test_empty_iterable(self):
        assert list(chunked([], 3)) == []

    def test_size_one(self):
        assert list(chunked([1, 2, 3], 1)) == [[1], [2], [3]]

    def test_invalid_size_zero(self):
        with pytest.raises(ValueError, match="chunk size must be"):
            list(chunked([1, 2], 0))

    def test_invalid_size_negative(self):
        with pytest.raises(ValueError):
            list(chunked([1, 2], -1))

    def test_with_generator(self):
        gen = (x for x in range(5))
        assert list(chunked(gen, 2)) == [[0, 1], [2, 3], [4]]

    def test_with_strings(self):
        assert list(chunked("abcde", 2)) == [["a", "b"], ["c", "d"], ["e"]]


# ===========================================================================
#  deduplicate
# ===========================================================================

class TestDeduplicate:
    def test_basic(self):
        assert deduplicate([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]

    def test_preserves_order(self):
        assert deduplicate([3, 1, 2, 1, 3]) == [3, 1, 2]

    def test_strings(self):
        assert deduplicate(["b", "a", "b", "c"]) == ["b", "a", "c"]

    def test_empty(self):
        assert deduplicate([]) == []

    def test_no_duplicates(self):
        assert deduplicate([1, 2, 3]) == [1, 2, 3]

    def test_all_duplicates(self):
        assert deduplicate([1, 1, 1]) == [1]

    def test_with_key_function(self):
        data = [{"id": 1, "v": "a"}, {"id": 1, "v": "b"}, {"id": 2, "v": "c"}]
        result = deduplicate(data, key=lambda x: x["id"])
        assert result == [{"id": 1, "v": "a"}, {"id": 2, "v": "c"}]

    def test_with_unhashable_items_and_key(self):
        data = [[1, 2], [3, 4], [1, 2]]
        result = deduplicate(data, key=lambda x: tuple(x))
        assert result == [[1, 2], [3, 4]]


# ===========================================================================
#  invert_dict
# ===========================================================================

class TestInvertDict:
    def test_basic(self):
        assert invert_dict({"a": 1, "b": 2}) == {1: "a", 2: "b"}

    def test_int_keys(self):
        assert invert_dict({1: "x", 2: "y"}) == {"x": 1, "y": 2}

    def test_empty(self):
        assert invert_dict({}) == {}

    def test_duplicate_values_last_wins(self):
        assert invert_dict({"a": 1, "b": 1}) == {1: "b"}

    def test_original_untouched(self):
        d = {"a": 1, "b": 2}
        invert_dict(d)
        assert d == {"a": 1, "b": 2}


# ===========================================================================
#  group_by
# ===========================================================================

class TestGroupBy:
    def test_by_parity(self):
        assert group_by([1, 2, 3, 4, 5, 6], lambda x: x % 2) == {1: [1, 3, 5], 0: [2, 4, 6]}

    def test_by_first_char(self):
        result = group_by(["apple", "avocado", "banana"], lambda s: s[0])
        assert result == {"a": ["apple", "avocado"], "b": ["banana"]}

    def test_empty(self):
        assert group_by([], lambda x: x) == {}

    def test_single_group(self):
        assert group_by([1, 2, 3], lambda x: "all") == {"all": [1, 2, 3]}

    def test_preserves_order_within_groups(self):
        result = group_by([3, 1, 4, 1, 5], lambda x: x % 2)
        assert result[1] == [3, 1, 1, 5]
        assert result[0] == [4]

    def test_with_tuples_as_keys(self):
        data = [(1, "a"), (1, "b"), (2, "c")]
        result = group_by(data, lambda x: x[0])
        assert result == {1: [(1, "a"), (1, "b")], 2: [(2, "c")]}


# ===========================================================================
#  FrozenDict
# ===========================================================================

class TestFrozenDict:
    def test_basic_access(self):
        fd = FrozenDict({"x": 1, "y": 2})
        assert fd["x"] == 1
        assert fd["y"] == 2

    def test_len(self):
        assert len(FrozenDict({"a": 1, "b": 2})) == 2

    def test_contains(self):
        fd = FrozenDict({"a": 1})
        assert "a" in fd
        assert "b" not in fd

    def test_iteration(self):
        fd = FrozenDict({"a": 1, "b": 2})
        assert sorted(fd.keys()) == ["a", "b"]
        assert sorted(fd.values()) == [1, 2]

    def test_from_kwargs(self):
        fd = FrozenDict(a=1, b=2)
        assert fd["a"] == 1
        assert fd["b"] == 2

    def test_from_mapping_and_kwargs(self):
        fd = FrozenDict({"a": 1}, b=2)
        assert fd["a"] == 1
        assert fd["b"] == 2

    def test_hashable(self):
        fd = FrozenDict({"a": 1, "b": 2})
        assert hash(fd) == hash(FrozenDict({"a": 1, "b": 2}))

    def test_can_be_dict_key(self):
        d = {FrozenDict({"a": 1}): "value"}
        assert d[FrozenDict({"a": 1})] == "value"

    def test_can_be_set_member(self):
        s = {FrozenDict({"a": 1}), FrozenDict({"a": 1}), FrozenDict({"b": 2})}
        assert len(s) == 2

    def test_equality_same(self):
        assert FrozenDict({"a": 1}) == FrozenDict({"a": 1})

    def test_equality_different(self):
        assert FrozenDict({"a": 1}) != FrozenDict({"a": 2})

    def test_equality_with_plain_dict(self):
        assert FrozenDict({"a": 1}) == {"a": 1}

    def test_inequality_with_plain_dict(self):
        assert FrozenDict({"a": 1}) != {"a": 2}

    def test_repr(self):
        fd = FrozenDict({"a": 1})
        assert "FrozenDict" in repr(fd)
        assert "'a': 1" in repr(fd)

    def test_empty(self):
        fd = FrozenDict()
        assert len(fd) == 0
        assert list(fd) == []

    def test_setitem_raises(self):
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="does not support item assignment"):
            fd["b"] = 2

    def test_delitem_raises(self):
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="does not support item deletion"):
            del fd["a"]

    def test_clear_raises(self):
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            fd.clear()

    def test_pop_raises(self):
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            fd.pop("a")

    def test_popitem_raises(self):
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            fd.popitem()

    def test_update_raises(self):
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            fd.update({"b": 2})

    def test_setdefault_raises(self):
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            fd.setdefault("b", 2)

    def test_get_method_works(self):
        fd = FrozenDict({"a": 1})
        assert fd.get("a") == 1
        assert fd.get("b", "default") == "default"

    def test_items_method_works(self):
        fd = FrozenDict({"a": 1, "b": 2})
        assert sorted(fd.items()) == [("a", 1), ("b", 2)]

    def test_hash_consistency(self):
        fd1 = FrozenDict({"x": 10, "y": 20})
        fd2 = FrozenDict({"y": 20, "x": 10})
        assert hash(fd1) == hash(fd2)

    def test_ne_operator(self):
        assert FrozenDict({"a": 1}) != FrozenDict({"a": 2})
        assert FrozenDict({"a": 1}) != {"a": 2}