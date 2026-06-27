"""
Unit tests for the new text processing utilities inside ilovetools.text.
"""

import pytest
from ilovetools.text import (
    extract_code_blocks,
    recursive_character_splitter,
    repair_and_parse_json,
    truncate_text_middle,
)

# ==========================================
# 1. Tests for extract_code_blocks
# ==========================================

def test_extract_code_blocks_basic():
    text = "Some intro text.\n```python\nprint('hello')\n```\nAnd a JSON block:\n```json\n{\"key\": \"val\"}\n```"
    
    # Extract all blocks
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0] == "print('hello')"
    assert blocks[1] == '{"key": "val"}'

def test_extract_code_blocks_by_language():
    text = "Some intro text.\n```python\nprint('hello')\n```\nAnd a JSON block:\n```json\n{\"key\": \"val\"}\n```"
    
    # Extract only python
    py_blocks = extract_code_blocks(text, "python")
    assert len(py_blocks) == 1
    assert py_blocks[0] == "print('hello')"
    
    # Extract only json
    json_blocks = extract_code_blocks(text, "json")
    assert len(json_blocks) == 1
    assert json_blocks[0] == '{"key": "val"}'
    
    # Extract non-existent language
    cpp_blocks = extract_code_blocks(text, "cpp")
    assert len(cpp_blocks) == 0

def test_extract_code_blocks_empty_invalid():
    assert extract_code_blocks("") == []
    assert extract_code_blocks(None) == []
    assert extract_code_blocks("Just regular text with no backticks.") == []


# ==========================================
# 2. Tests for recursive_character_splitter
# ==========================================

def test_recursive_splitter_basic():
    text = "Hello world! This is a test string to check the recursive character splitter."
    # Max chunk size of 15, no overlap
    chunks = recursive_character_splitter(text, chunk_size=15, chunk_overlap=0)
    for chunk in chunks:
        assert len(chunk) <= 15
    assert len(chunks) > 1

def test_recursive_splitter_with_overlap():
    text = "Paragraph one of text.\n\nParagraph two with more text.\n\nParagraph three."
    chunks = recursive_character_splitter(text, chunk_size=30, chunk_overlap=5)
    for chunk in chunks:
        assert len(chunk) <= 30
    assert len(chunks) > 1

def test_recursive_splitter_exceptions():
    with pytest.raises(ValueError):
        recursive_character_splitter("Hello", chunk_size=0)
    with pytest.raises(ValueError):
        recursive_character_splitter("Hello", chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ValueError):
        recursive_character_splitter("Hello", chunk_size=10, chunk_overlap=11)


# ==========================================
# 3. Tests for repair_and_parse_json
# ==========================================

def test_repair_and_parse_json_clean():
    raw = '{"name": "Ali", "role": "developer"}'
    assert repair_and_parse_json(raw) == {"name": "Ali", "role": "developer"}

def test_repair_and_parse_json_markdown():
    raw = 'Sure, here is your JSON response:\n```json\n{\n  "status": "success",\n  "code": 200\n}\n```\nLet me know if you need anything else.'
    assert repair_and_parse_json(raw) == {"status": "success", "code": 200}

def test_repair_and_parse_json_trailing_commas():
    raw = '{\n  "skills": ["python", "ai",],\n  "details": {\n    "active": true,\n  },\n}'
    assert repair_and_parse_json(raw) == {"skills": ["python", "ai"], "details": {"active": True}}

def test_repair_and_parse_json_comments():
    raw = """
    // Config values
    {
      "port": 8080, # web server port
      "host": "localhost" // local host bind
    }
    """
    assert repair_and_parse_json(raw) == {"port": 8080, "host": "localhost"}

def test_repair_and_parse_json_single_quotes():
    raw = "{'user_id': 123, 'tags': ['admin', 'pro']}"
    assert repair_and_parse_json(raw) == {"user_id": 123, "tags": ["admin", "pro"]}

def test_repair_and_parse_json_errors():
    with pytest.raises(TypeError):
        repair_and_parse_json(1234)
    with pytest.raises(ValueError):
        repair_and_parse_json("not json at all")


# ==========================================
# 4. Tests for truncate_text_middle
# ==========================================

def test_truncate_text_middle_basic():
    text = "abcdefghijklmnopqrstuvwxyz"
    assert truncate_text_middle(text, 10) == "abc...wxyz"
    assert truncate_text_middle(text, 26) == text
    assert truncate_text_middle(text, 50) == text

def test_truncate_text_middle_custom_placeholder():
    text = "abcdefghijklmnopqrstuvwxyz"
    assert truncate_text_middle(text, 10, placeholder="--") == "abcd--wxyz"

def test_truncate_text_middle_small_max():
    text = "abcdef"
    assert truncate_text_middle(text, 2) == ".."
    assert truncate_text_middle(text, 0) == ""
    assert truncate_text_middle(None, 10) == ""
