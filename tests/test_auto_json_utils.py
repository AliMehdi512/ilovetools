"""
Comprehensive pytest suite for ilovetools.utils.json_utils.

Tests cover all public functions, edge cases, error handling, and
typical LLM-agent usage patterns.
"""

import json
import pytest
from ilovetools.utils.json_utils import (
    extract_json,
    repair_json,
    safe_json_loads,
    merge_json,
    diff_json,
    flatten_json,
    unflatten_json,
    json_path_get,
    json_path_set,
    validate_json_schema,
    redact_json_keys,
    json_size,
)


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_extract_object(self):
        text = 'noise {"name": "Ali", "age": 30} noise'
        assert extract_json(text) == {"name": "Ali", "age": 30}

    def test_extract_array(self):
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_extract_from_markdown_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert extract_json(text) == {"a": 1}

    def test_extract_first_json_in_mixed(self):
        # When both { and [ are present, { is searched first
        text = 'text [1, 2] more {"a": 1}'
        result = extract_json(text)
        # The function searches for { first, so {"a": 1} is found
        assert result == {"a": 1}

    def test_extract_array_before_object(self):
        # Array appears first in text and no { before it
        text = 'text [1, 2] more text'
        assert extract_json(text) == [1, 2]

    def test_extract_no_json(self):
        assert extract_json("no json here") is None

    def test_extract_empty_string(self):
        assert extract_json("") is None

    def test_extract_none_input(self):
        assert extract_json(None) is None

    def test_extract_nested_object(self):
        text = 'prefix {"a": {"b": [1, 2]}} suffix'
        assert extract_json(text) == {"a": {"b": [1, 2]}}

    def test_extract_direct_parse(self):
        assert extract_json('{"x": 42}') == {"x": 42}


# ---------------------------------------------------------------------------
# repair_json
# ---------------------------------------------------------------------------

class TestRepairJson:
    def test_single_quotes(self):
        assert repair_json("{'a': 1, 'b': 2,}") == '{"a": 1, "b": 2}'

    def test_unquoted_keys(self):
        assert repair_json('{key: "value"}') == '{"key": "value"}'

    def test_python_booleans(self):
        result = repair_json('{a: True, b: False}')
        assert "true" in result
        assert "false" in result

    def test_python_none(self):
        result = repair_json('{a: None}')
        assert "null" in result

    def test_markdown_fences(self):
        assert repair_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_trailing_comma_in_object(self):
        result = repair_json('{"a": 1, "b": 2,}')
        assert ",}" not in result

    def test_trailing_comma_in_array(self):
        result = repair_json('[1, 2, 3,]')
        assert ",]" not in result

    def test_comments_removed(self):
        result = repair_json('{"a": 1 // comment\n}')
        assert "//" not in result

    def test_multiline_comments_removed(self):
        result = repair_json('{"a": 1 /* comment */}')
        assert "/*" not in result

    def test_ellipsis_removed(self):
        result = repair_json('{"a": 1, ...}')
        assert "..." not in result

    def test_non_string_input(self):
        assert repair_json(123) == 123


# ---------------------------------------------------------------------------
# safe_json_loads
# ---------------------------------------------------------------------------

class TestSafeJsonLoads:
    def test_valid_json(self):
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_repairable_json(self):
        assert safe_json_loads("{'a': 1,}") == {"a": 1}

    def test_completely_invalid(self):
        assert safe_json_loads("not json at all") is None

    def test_empty_string(self):
        assert safe_json_loads("") is None

    def test_none_input(self):
        assert safe_json_loads(None) is None

    def test_no_repair_flag(self):
        # Without repair, malformed JSON should fail
        assert safe_json_loads("{'a': 1}", repair=False) is None

    def test_extract_fallback(self):
        result = safe_json_loads('Here is data: {"x": 42}')
        assert result == {"x": 42}


# ---------------------------------------------------------------------------
# merge_json
# ---------------------------------------------------------------------------

class TestMergeJson:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"c": 3}
        assert merge_json(base, override) == {"a": 1, "b": 2, "c": 3}

    def test_deep_merge(self):
        base = {"a": 1, "b": {"x": 1}}
        override = {"b": {"y": 2}, "c": 3}
        assert merge_json(base, override) == {"a": 1, "b": {"x": 1, "y": 2}, "c": 3}

    def test_override_replaces_non_dict(self):
        base = {"a": 1}
        override = {"a": "hello"}
        assert merge_json(base, override) == {"a": "hello"}

    def test_list_replaced_not_merged(self):
        base = {"a": [1, 2]}
        override = {"a": [3]}
        assert merge_json(base, override) == {"a": [3]}

    def test_empty_base(self):
        assert merge_json({}, {"x": 0}) == {"x": 0}

    def test_empty_override(self):
        assert merge_json({"x": 0}, {}) == {"x": 0}

    def test_inputs_not_mutated(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        merge_json(base, override)
        assert base == {"a": {"x": 1}}


# ---------------------------------------------------------------------------
# diff_json
# ---------------------------------------------------------------------------

class TestDiffJson:
    def test_added_key(self):
        result = diff_json({"a": 1}, {"a": 1, "b": 2})
        assert result["added"] == {"b": 2}
        assert result["removed"] == {}
        assert result["changed"] == {}

    def test_removed_key(self):
        result = diff_json({"a": 1, "b": 2}, {"a": 1})
        assert result["removed"] == {"b": 2}
        assert result["added"] == {}

    def test_changed_value(self):
        result = diff_json({"a": 1}, {"a": 2})
        assert result["changed"] == {"a": {"old": 1, "new": 2}}

    def test_no_changes(self):
        result = diff_json({"a": 1}, {"a": 1})
        assert result == {"added": {}, "removed": {}, "changed": {}}

    def test_multiple_changes(self):
        result = diff_json({"a": 1, "b": 2, "c": 3}, {"a": 1, "c": 33, "d": 4})
        assert result["added"] == {"d": 4}
        assert result["removed"] == {"b": 2}
        assert result["changed"] == {"c": {"old": 3, "new": 33}}


# ---------------------------------------------------------------------------
# flatten_json / unflatten_json
# ---------------------------------------------------------------------------

class TestFlattenUnflatten:
    def test_flatten_nested(self):
        assert flatten_json({"a": {"b": {"c": 1}}}) == {"a.b.c": 1}

    def test_flatten_with_list(self):
        result = flatten_json({"a": [1, 2]})
        assert result == {"a.0": 1, "a.1": 2}

    def test_flatten_empty(self):
        assert flatten_json({}) == {}

    def test_flatten_flat(self):
        assert flatten_json({"x": 1}) == {"x": 1}

    def test_unflatten_simple(self):
        assert unflatten_json({"a.b.c": 1}) == {"a": {"b": {"c": 1}}}

    def test_unflatten_mixed(self):
        assert unflatten_json({"x": 1, "y.z": 2}) == {"x": 1, "y": {"z": 2}}

    def test_unflatten_empty(self):
        assert unflatten_json({}) == {}

    def test_roundtrip(self):
        original = {"a": {"b": {"c": 1}}, "d": 2}
        flat = flatten_json(original)
        restored = unflatten_json(flat)
        assert restored == original


# ---------------------------------------------------------------------------
# json_path_get / json_path_set
# ---------------------------------------------------------------------------

class TestJsonPath:
    def test_get_nested(self):
        assert json_path_get({"a": {"b": {"c": 42}}}, "a.b.c") == 42

    def test_get_missing(self):
        assert json_path_get({}, "missing.path") is None

    def test_get_from_list(self):
        assert json_path_get({"x": [10, 20]}, "x.1") == 20

    def test_get_empty_path_raises(self):
        with pytest.raises(KeyError):
            json_path_get({"a": 1}, "")

    def test_set_nested(self):
        data = {}
        json_path_set(data, "x.y.z", 99)
        assert data == {"x": {"y": {"z": 99}}}

    def test_set_existing(self):
        data = {"a": {"b": 1}}
        json_path_set(data, "a.c", 2)
        assert data == {"a": {"b": 1, "c": 2}}

    def test_set_empty_path_raises(self):
        with pytest.raises(KeyError):
            json_path_set({}, "", 1)


# ---------------------------------------------------------------------------
# validate_json_schema
# ---------------------------------------------------------------------------

class TestValidateJsonSchema:
    def test_valid_object(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "str"},
                "age": {"type": "int"},
            },
        }
        assert validate_json_schema({"name": "Ali", "age": 30}, schema) is True

    def test_missing_required(self):
        schema = {"type": "object", "required": ["name"]}
        assert validate_json_schema({"age": 30}, schema) is False

    def test_wrong_type(self):
        schema = {"type": "str"}
        assert validate_json_schema(123, schema) is False

    def test_correct_type_str(self):
        assert validate_json_schema("hello", {"type": "str"}) is True

    def test_correct_type_int(self):
        assert validate_json_schema(42, {"type": "int"}) is True

    def test_bool_not_int(self):
        assert validate_json_schema(True, {"type": "int"}) is False

    def test_minimum(self):
        assert validate_json_schema(5, {"type": "int", "minimum": 0}) is True
        assert validate_json_schema(-1, {"type": "int", "minimum": 0}) is False

    def test_maximum(self):
        assert validate_json_schema(5, {"type": "int", "maximum": 10}) is True
        assert validate_json_schema(15, {"type": "int", "maximum": 10}) is False

    def test_enum_valid(self):
        assert validate_json_schema("a", {"enum": ["a", "b", "c"]}) is True

    def test_enum_invalid(self):
        assert validate_json_schema("d", {"enum": ["a", "b", "c"]}) is False

    def test_array_items(self):
        schema = {"type": "list", "items": {"type": "int"}}
        assert validate_json_schema([1, 2, 3], schema) is True
        assert validate_json_schema([1, "two", 3], schema) is False

    def test_no_schema(self):
        assert validate_json_schema("anything", {}) is True

    def test_nested_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {"id": {"type": "int"}},
                },
            },
        }
        assert validate_json_schema({"user": {"id": 1}}, schema) is True
        assert validate_json_schema({"user": {}}, schema) is False


# ---------------------------------------------------------------------------
# redact_json_keys
# ---------------------------------------------------------------------------

class TestRedactJsonKeys:
    def test_default_redaction(self):
        data = {"user": "Ali", "api_key": "secret123"}
        result = redact_json_keys(data)
        assert result["api_key"] == "[REDACTED]"
        assert result["user"] == "Ali"

    def test_nested_redaction(self):
        data = {"data": {"token": "xyz"}}
        result = redact_json_keys(data)
        assert result["data"]["token"] == "[REDACTED]"

    def test_custom_keys(self):
        data = {"secret_value": "abc", "safe": "ok"}
        result = redact_json_keys(data, redact_keys=["secret_value"])
        assert result["secret_value"] == "[REDACTED]"
        assert result["safe"] == "ok"

    def test_list_redaction(self):
        data = [{"password": "abc"}, {"safe": "ok"}]
        result = redact_json_keys(data)
        assert result[0]["password"] == "[REDACTED]"
        assert result[1]["safe"] == "ok"

    def test_custom_placeholder(self):
        data = {"token": "xyz"}
        result = redact_json_keys(data, placeholder="***")
        assert result["token"] == "***"

    def test_case_insensitive(self):
        data = {"API_KEY": "secret"}
        result = redact_json_keys(data)
        assert result["API_KEY"] == "[REDACTED]"

    def test_non_dict_input(self):
        assert redact_json_keys("hello") == "hello"
        assert redact_json_keys(42) == 42


# ---------------------------------------------------------------------------
# json_size
# ---------------------------------------------------------------------------

class TestJsonSize:
    def test_simple_object(self):
        assert json_size({"a": 1}) == 7  # {"a":1}

    def test_array(self):
        assert json_size([1, 2, 3]) == 7  # [1,2,3]

    def test_string(self):
        assert json_size("hello") == 7  # "hello"

    def test_none(self):
        assert json_size(None) == 4  # null

    def test_empty_object(self):
        assert json_size({}) == 2  # {}

    def test_nested(self):
        size = json_size({"a": {"b": [1, 2]}})
        assert size > 0
        # Verify it matches json.dumps
        expected = len(json.dumps({"a": {"b": [1, 2]}}, separators=(",", ":")).encode("utf-8"))
        assert size == expected
