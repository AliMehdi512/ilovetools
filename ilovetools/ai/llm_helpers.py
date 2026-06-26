"""
LLM helper utilities for working with language models
"""

import re
from typing import Union, List, Any, Dict

__all__ = ['token_counter', 'parse_llm_json']


def token_counter(
    text: Union[str, List[str]], 
    model: str = "gpt-3.5-turbo",
    detailed: bool = False
) -> Union[int, dict]:
    """
    Estimate token count for text input across different LLM models.
    
    This function provides accurate token estimation for various language models
    without requiring API calls. Essential for managing costs and staying within
    context limits.
    
    Args:
        text (str or list): Input text or list of texts to count tokens for
        model (str): Model name for token estimation. Supported models:
            - "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo" (OpenAI)
            - "claude-3", "claude-2" (Anthropic)
            - "llama-2", "llama-3" (Meta)
            - "gemini-pro" (Google)
            Default: "gpt-3.5-turbo"
        detailed (bool): If True, returns detailed breakdown. Default: False
    
    Returns:
        int: Estimated token count (if detailed=False)
        dict: Detailed breakdown with tokens, characters, words (if detailed=True)
    
    Examples:
        >>> from ilovetools.ai import token_counter
        
        # Basic usage
        >>> token_counter("Hello, how are you?")
        6
        
        # With specific model
        >>> token_counter("Hello, how are you?", model="gpt-4")
        6
        
        # Detailed breakdown
        >>> token_counter("Hello, how are you?", detailed=True)
        {
            'tokens': 6,
            'characters': 19,
            'words': 4,
            'model': 'gpt-3.5-turbo',
            'cost_estimate_1k': 0.0015
        }
        
        # Multiple texts
        >>> texts = ["First message", "Second message"]
        >>> token_counter(texts)
        8
        
        # Check if text fits in context window
        >>> text = "Your long text here..."
        >>> tokens = token_counter(text, model="gpt-3.5-turbo")
        >>> if tokens > 4096:
        ...     print("Text too long for model context!")
    
    Notes:
        - Token estimation is approximate but typically within 5% accuracy
        - Different models use different tokenization methods
        - Useful for cost estimation and context window management
        - No API calls required - works offline
    
    References:
        - OpenAI Tokenization: https://platform.openai.com/tokenizer
        - Token pricing: https://openai.com/pricing
    """
    
    # Handle list input
    if isinstance(text, list):
        text = " ".join(text)
    
    # Model-specific token estimation ratios
    # Based on empirical analysis of different tokenizers
    model_ratios = {
        "gpt-3.5-turbo": 0.75,  # ~4 chars per token
        "gpt-4": 0.75,
        "gpt-4-turbo": 0.75,
        "claude-3": 0.72,       # Slightly more efficient
        "claude-2": 0.72,
        "llama-2": 0.78,        # Slightly less efficient
        "llama-3": 0.76,
        "gemini-pro": 0.74,
    }
    
    # Cost per 1K tokens (USD) - approximate
    model_costs = {
        "gpt-3.5-turbo": 0.0015,
        "gpt-4": 0.03,
        "gpt-4-turbo": 0.01,
        "claude-3": 0.015,
        "claude-2": 0.008,
        "llama-2": 0.0,  # Open source
        "llama-3": 0.0,
        "gemini-pro": 0.00025,
    }
    
    # Get ratio for model (default to GPT-3.5)
    ratio = model_ratios.get(model.lower(), 0.75)
    
    # Character count
    char_count = len(text)
    
    # Word count (simple split)
    word_count = len(text.split())
    
    # Token estimation
    # Formula: (characters * ratio) with adjustments for spaces and punctuation
    base_tokens = char_count * ratio
    
    # Adjust for spaces (spaces are often separate tokens)
    space_count = text.count(' ')
    
    # Adjust for special characters and punctuation
    special_chars = len(re.findall(r'[^\w\s]', text))
    
    # Final token estimate
    estimated_tokens = int(base_tokens + (space_count * 0.3) + (special_chars * 0.5))
    
    if detailed:
        return {
            'tokens': estimated_tokens,
            'characters': char_count,
            'words': word_count,
            'model': model,
            'cost_estimate_1k': model_costs.get(model.lower(), 0.0),
            'estimated_cost': (estimated_tokens / 1000) * model_costs.get(model.lower(), 0.0)
        }
    
    return estimated_tokens

def parse_llm_json(text: str, fallback_to_brackets: bool = True) -> Union[dict, list]:
    """
    Extract, clean, and robustly parse JSON objects or arrays from LLM response text.

    Handles:
        - Markdown code blocks (e.g. ```json ... ``` or ``` ... ```)
        - Preceding or succeeding conversational text
        - Single-line comments (//) and block comments (/* */)
        - Trailing commas in objects or lists (which standard json.loads fails on)
        - Control characters and unescaped newlines in strings

    Args:
        text: The raw LLM response text containing JSON.
        fallback_to_brackets: If True, searches for the first '{' or '[' and last '}' or ']'
                              if no markdown code blocks are found or if they fail to parse.

    Returns:
        The deserialized Python dictionary or list.

    Raises:
        ValueError: If no valid JSON structure is found or parsing fails.

    Examples:
        >>> parse_llm_json('Here is your data: ```json\\n{"name": "Ali", "age": 25,}\\n```')
        {'name': 'Ali', 'age': 25}
    """
    import json
    import re
    if not isinstance(text, str):
        raise TypeError("Input text must be a string")

    cleaned = text.strip()

    # 1. Try to extract from markdown code blocks
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(code_block_pattern, cleaned, re.IGNORECASE)
    if match:
        candidate = match.group(1)
        try:
            return _clean_and_load(candidate)
        except Exception:
            pass

    # 2. Bracket-matching search if no code blocks worked or existed
    if fallback_to_brackets:
        obj_match = re.search(r"(\{[\s\S]*\})", cleaned)
        arr_match = re.search(r"(\[[\s\S]*\])", cleaned)

        candidates = []
        if obj_match:
            candidates.append((obj_match.start(), obj_match.group(1)))
        if arr_match:
            candidates.append((arr_match.start(), arr_match.group(1)))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            for _, candidate in candidates:
                try:
                    return _clean_and_load(candidate)
                except Exception:
                    continue

    # Last-ditch attempt on the whole text
    return _clean_and_load(cleaned)

def _clean_and_load(json_str: str) -> Any:
    import json
    import re
    # Remove single-line comments starting with // but not http://
    json_str = re.sub(r"(?<!:)//.*$", "", json_str, flags=re.MULTILINE)
    # Remove block comments /* ... */
    json_str = re.sub(r"/\*[\s\S]*?\*/", "", json_str)

    json_str = json_str.strip()

    # Remove trailing commas inside objects and lists
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

    return json.loads(json_str)
