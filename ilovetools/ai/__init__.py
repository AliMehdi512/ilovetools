"""
AI & Machine Learning utilities module
"""

from .llm_helpers import token_counter, parse_llm_json
from .embeddings import similarity_search, cosine_similarity
from .prompt_engineering import (
    PromptBuilder,
    PromptTemplate,
    build_few_shot_prompt,
    extract_variables,
    fill_template,
    truncate_for_context,
    estimate_api_cost,
    format_chat_messages,
    MODEL_CONTEXT_WINDOWS,
    MODEL_PRICING,
)
from .inference import *
from .context_manager import (
    ContextWindow,
    ConversationBuffer,
    ConversationSummaryBuffer,
    trim_messages,
    summarize_history,
    allocate_token_budget,
    count_message_tokens,
    estimate_tokens,
)
from .tool_registry import (
    ToolRegistry,
    Tool,
    tool,
    ToolError,
    ToolResult,
)

__all__ = [
    'token_counter',
    'parse_llm_json',
    'similarity_search',
    'cosine_similarity',
    'PromptBuilder',
    'PromptTemplate',
    'build_few_shot_prompt',
    'extract_variables',
    'fill_template',
    'truncate_for_context',
    'estimate_api_cost',
    'format_chat_messages',
    'MODEL_CONTEXT_WINDOWS',
    'MODEL_PRICING',
    # Context Manager
    'ContextWindow',
    'ConversationBuffer',
    'ConversationSummaryBuffer',
    'trim_messages',
    'summarize_history',
    'allocate_token_budget',
    'count_message_tokens',
    'estimate_tokens',
    # Tool Registry
    'ToolRegistry',
    'Tool',
    'tool',
    'ToolError',
    'ToolResult',
]
