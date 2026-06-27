"""
Core text processing utilities for developer workflows and LLM agent integration.
"""

import re
import json
import ast
from typing import List, Optional, Any, Dict, Union

def extract_code_blocks(text: str, language: Optional[str] = None) -> List[str]:
    """
    Extracts markdown code blocks enclosed in triple backticks (```).
    
    If a specific language is provided, only extracts blocks matching that language
    (case-insensitive, matching the identifier right after the triple backticks).
    
    Args:
        text: The source text containing markdown code blocks.
        language: Optional language identifier to filter by (e.g., 'python', 'json').
                  If None, all code blocks are returned.
                  
    Returns:
        A list of extracted code block contents with leading/trailing whitespace stripped.
        
    Examples:
        >>> text = "Hello\\n```python\\nprint('hello')\\n```\\nWorld\\n```json\\n{\\"a\\": 1}\\n```"
        >>> extract_code_blocks(text, 'python')
        ["print('hello')"]
        >>> extract_code_blocks(text)
        ["print('hello')", '{\\n  "a": 1\\n}']
    """
    if not isinstance(text, str) or not text:
        return []
        
    # Matches ```[language]\n[code_content]\n``` with flexible trailing/leading spacing
    pattern = r"```([a-zA-Z0-9_\-\+]*)\s*\n(.*?)\n\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    results = []
    for lang_match, code_content in matches:
        lang_match = lang_match.strip().lower()
        if language is not None:
            if lang_match != language.strip().lower():
                continue
        results.append(code_content.strip())
    return results

def recursive_character_splitter(
    text: str, 
    chunk_size: int, 
    chunk_overlap: int = 0, 
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Recursively splits a text into smaller chunks of at most chunk_size characters,
    trying to keep semantically coherent units together using a list of separators.
    
    Args:
        text: The input text to split.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The amount of character overlap between consecutive chunks.
        separators: An ordered list of separator strings to try splitting on.
                    Defaults to ["\\n\\n", "\\n", " ", ""].
                    
    Returns:
        A list of split text chunks.
        
    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size.
    """
    if not isinstance(text, str):
        return []
        
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size.")
        
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
        
    def _split_text(txt: str, current_seps: List[str]) -> List[str]:
        if len(txt) <= chunk_size:
            return [txt]
            
        if not current_seps:
            # No separators left, hard split by chunk_size
            return [txt[i:i + chunk_size] for i in range(0, len(txt), chunk_size - chunk_overlap)]
            
        separator = current_seps[0]
        remaining_seps = current_seps[1:]
        
        # Split by the separator
        if separator == "":
            splits = list(txt)
        else:
            splits = txt.split(separator)
            
        chunks = []
        current_chunk = []
        current_len = 0
        
        for part in splits:
            # Add back the separator if it's not the first element (except for empty separator)
            part_str = part if (not current_chunk or separator == "") else (separator + part)
            part_len = len(part_str)
            
            if current_len + part_len <= chunk_size:
                current_chunk.append(part_str)
                current_len += part_len
            else:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                
                # If the single part itself is larger than chunk_size, we must split it recursively
                if len(part) > chunk_size:
                    chunks.extend(_split_text(part, remaining_seps))
                    current_chunk = []
                    current_len = 0
                else:
                    current_chunk = [part]
                    current_len = len(part)
                    
        if current_chunk:
            chunks.append("".join(current_chunk))
            
        return chunks

    raw_chunks = _split_text(text, separators)
    
    if chunk_overlap == 0:
        return [c.strip() for c in raw_chunks if c.strip()]
        
    # Apply sliding window overlap to raw_chunks
    overlapped_chunks = []
    current_doc = ""
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if not current_doc:
            current_doc = chunk
        elif len(current_doc) + 1 + len(chunk) <= chunk_size:
            current_doc += " " + chunk
        else:
            overlapped_chunks.append(current_doc)
            # Find overlap suffix
            overlap_start = max(0, len(current_doc) - chunk_overlap)
            overlap_part = current_doc[overlap_start:]
            current_doc = (overlap_part + " " + chunk) if overlap_part else chunk
            if len(current_doc) > chunk_size:
                current_doc = current_doc[-chunk_size:]
                
    if current_doc:
        overlapped_chunks.append(current_doc)
        
    return [c.strip() for c in overlapped_chunks if c.strip()]

def repair_and_parse_json(text: str) -> Any:
    """
    Cleans up a dirty string (e.g. LLM response with conversational prefix/suffix,
    markdown wrapping, trailing commas, single-quoted keys/values, or Python style comments) 
    and parses it into a Python dict or list.
    
    Args:
        text: The dirty JSON-like string to clean and parse.
        
    Returns:
        The parsed Python object (dict or list).
        
    Raises:
        TypeError: If input text is not a string.
        ValueError: If JSON parsing and ast literal parsing both fail.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")
        
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Cannot parse an empty string as JSON.")
        
    # 1. If wrapped in markdown code blocks, extract it
    code_blocks = extract_code_blocks(cleaned, "json")
    if code_blocks:
        cleaned = code_blocks[0].strip()
    else:
        # Check any markdown code block if no 'json' tag was matched
        any_blocks = extract_code_blocks(cleaned)
        if any_blocks:
            cleaned = any_blocks[0].strip()
            
    # 2. Extract JSON portion if surrounded by conversational filler
    first_brace = cleaned.find('{')
    first_bracket = cleaned.find('[')
    
    start_idx = -1
    end_char = ''
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_idx = first_brace
        end_char = '}'
    elif first_bracket != -1:
        start_idx = first_bracket
        end_char = ']'
        
    if start_idx != -1:
        end_idx = cleaned.rfind(end_char)
        if end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx:end_idx + 1]
            
    # 3. Strip single-line comments (starting with // or #)
    lines = []
    for line in cleaned.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith('//') or stripped_line.startswith('#'):
            continue
        if '//' in line and 'http://' not in line and 'https://' not in line:
            line = line.split('//', 1)[0]
        elif '#' in line and '"' not in line and "'" not in line:
            line = line.split('#', 1)[0]
        lines.append(line)
    cleaned = "\n".join(lines)
    
    # 4. Remove trailing commas in objects or arrays
    cleaned = re.sub(r',\s*\}', '}', cleaned)
    cleaned = re.sub(r',\s*\]', ']', cleaned)
    
    # 5. Fix single quotes to double quotes for keys/values
    # Replace single quotes around keys: 'key': -> "key":
    cleaned = re.sub(r"'([a-zA-Z0-9_\-]+)'\s*:", r'"\1":', cleaned)
    # Replace single quotes around values: : 'value' -> : "value"
    cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)
    # Replace single quotes in lists: ['a', 'b'] -> ["a", "b"]
    cleaned = re.sub(r"\[\s*'([^']*)'", r'["\1"', cleaned)
    cleaned = re.sub(r"'\s*\]", r'"]', cleaned)
    cleaned = re.sub(r"'\s*,\s*'", r'", "', cleaned)
    cleaned = re.sub(r"'\s*,\s*\"", r'", "', cleaned)
    cleaned = re.sub(r"\"\s*,\s*'", r'", "', cleaned)
    
    # Try parsing
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Fall back to Python literal evaluation
        try:
            return ast.literal_eval(cleaned)
        except Exception as eval_err:
            raise ValueError(
                f"Failed to repair and parse JSON.\n"
                f"JSON loads error: {e}\n"
                f"AST evaluation error: {eval_err}\n"
                f"Cleaned text was:\n{cleaned}"
            )

def truncate_text_middle(text: str, max_chars: int, placeholder: str = "...") -> str:
    """
    Truncates a text string in the middle to a maximum of max_chars,
    inserting a placeholder string in between the prefix and suffix.
    
    Args:
        text: The string to truncate.
        max_chars: The maximum length of the resulting string.
        placeholder: The string to insert in the middle of truncation.
                     Defaults to "...".
                     
    Returns:
        The truncated string or the original string if its length is already
        less than or equal to max_chars.
    """
    if not isinstance(text, str):
        return ""
        
    if max_chars <= 0:
        return ""
        
    if len(text) <= max_chars:
        return text
        
    pl_len = len(placeholder)
    if max_chars <= pl_len:
        return placeholder[:max_chars]
        
    remaining_chars = max_chars - pl_len
    prefix_len = remaining_chars // 2
    suffix_len = remaining_chars - prefix_len
    
    prefix = text[:prefix_len]
    suffix = text[-suffix_len:] if suffix_len > 0 else ""
    
    return f"{prefix}{placeholder}{suffix}"
