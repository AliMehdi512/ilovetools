"""
Unit tests for the parse_llm_json helper utility.
"""

import pytest
from ilovetools.ai.llm_helpers import parse_llm_json

def test_parse_llm_json_clean():
    raw = '{"name": "Ali", "role": "developer"}'
    res = parse_llm_json(raw)
    assert res == {"name": "Ali", "role": "developer"}

def test_parse_llm_json_markdown():
    raw = 'Sure! Here is the JSON:\n```json\n{\n  "status": "success",\n  "code": 200\n}\n```\nHope this helps!'
    res = parse_llm_json(raw)
    assert res == {"status": "success", "code": 200}

def test_parse_llm_json_trailing_commas():
    raw = '{\n  "skills": ["python", "ai",],\n  "details": {\n    "active": true,\n  },\n}'
    res = parse_llm_json(raw)
    assert res == {"skills": ["python", "ai"], "details": {"active": True}}

def test_parse_llm_json_with_comments():
    raw = '''
    // Configuration file
    {
      "port": 8080, // Default web port
      /* Database configurations
         for dev environment */
      "db": "postgres"
    }
    '''
    res = parse_llm_json(raw)
    assert res == {"port": 8080, "db": "postgres"}

def test_parse_llm_json_invalid_types():
    with pytest.raises(TypeError):
        parse_llm_json(123)
