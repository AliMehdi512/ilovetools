"""
General utility functions
"""

from .rate_limiter import (
    TokenBucketLimiter,
    LeakyBucketLimiter,
    FixedWindowLimiter,
    SlidingWindowLimiter,
    MultiTierLimiter,
    AdaptiveRateLimiter,
    create_rate_limiter,
    RateLimitDecorator,
)

from .cache_system import (
    MemoryCache,
    LRUCache,
    TTLCache,
    FileCache,
    cache_decorator,
    memoize,
    clear_all_caches,
    CacheStats,
)

from .logger import (
    Logger,
    LogLevel,
    JSONFormatter,
    ColoredFormatter,
    StructuredLogger,
    create_logger,
    log_execution_time,
    log_errors,
)

from .retry import (
    retry,
    exponential_backoff,
    linear_backoff,
    constant_backoff,
    RetryStrategy,
    CircuitBreaker,
    retry_with_circuit_breaker,
)

from .data_structures import (
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

from .patterns import (
    Singleton,
    Observer,
    Strategy,
    ChainOfResponsibility,
    Command,
    CommandHistory,
    Registry,
    Pipeline,
    Builder,
)

from .async_helpers import (
    gather_with_limit,
    async_map,
    async_filter,
    async_retry,
    AsyncPool,
    race,
    async_timeout,
    AsyncCache,
)

from .json_utils import (
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

__all__ = [
    # Rate Limiter
    'TokenBucketLimiter',
    'LeakyBucketLimiter',
    'FixedWindowLimiter',
    'SlidingWindowLimiter',
    'MultiTierLimiter',
    'AdaptiveRateLimiter',
    'create_rate_limiter',
    'RateLimitDecorator',
    # Cache System
    'MemoryCache',
    'LRUCache',
    'TTLCache',
    'FileCache',
    'cache_decorator',
    'memoize',
    'clear_all_caches',
    'CacheStats',
    # Logger
    'Logger',
    'LogLevel',
    'JSONFormatter',
    'ColoredFormatter',
    'StructuredLogger',
    'create_logger',
    'log_execution_time',
    'log_errors',
    # Retry
    'retry',
    'exponential_backoff',
    'linear_backoff',
    'constant_backoff',
    'RetryStrategy',
    'CircuitBreaker',
    'retry_with_circuit_breaker',
    # Data Structures
    'deep_merge',
    'flatten_dict',
    'unflatten_dict',
    'deep_get',
    'deep_set',
    'deep_delete',
    'chunked',
    'deduplicate',
    'invert_dict',
    'group_by',
    'FrozenDict',
    # Design Patterns
    'Singleton',
    'Observer',
    'Strategy',
    'ChainOfResponsibility',
    'Command',
    'CommandHistory',
    'Registry',
    'Pipeline',
    'Builder',
    # Async Helpers
    'gather_with_limit',
    'async_map',
    'async_filter',
    'async_retry',
    'AsyncPool',
    'race',
    'async_timeout',
    'AsyncCache',
    # JSON Utils
    'extract_json',
    'repair_json',
    'safe_json_loads',
    'merge_json',
    'diff_json',
    'flatten_json',
    'unflatten_json',
    'json_path_get',
    'json_path_set',
    'validate_json_schema',
    'redact_json_keys',
    'json_size',
]
