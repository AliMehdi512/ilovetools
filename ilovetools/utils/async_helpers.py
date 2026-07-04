"""
Async concurrency utilities for modern Python developer and LLM-agent workflows.

This module provides a small, focused collection of async helpers that fill the
gaps between :mod:`asyncio` primitives and the patterns developers actually need
on a daily basis: bounded-concurrency ``gather``, async ``map`` / ``filter``,
retry for coroutines, a reusable worker pool, a "race" that returns the first
successful result, a timeout decorator, and a thread-safe async cache.

Every function and class is pure-Python, fully type-hinted, zero-dependency
(only the standard library), and safe to drop into any ``asyncio``-based
project — web scrapers, LLM API fan-out, batch processing, or agent tool
orchestration.

Quick start
-----------
.. code-block:: python

    import asyncio
    from ilovetools.utils.async_helpers import (
        gather_with_limit, async_map, async_filter, async_retry,
        AsyncPool, race, async_timeout, AsyncCache,
    )

    async def fetch(url: str) -> str:
        ...  # your HTTP / LLM call here

    # 1. Bounded concurrency – never more than 10 in-flight at once
    results = await gather_with_limit(fetch(u) for u in urls, limit=10)

    # 2. Async map with concurrency control
    pages = await async_map(fetch, urls, limit=5)

    # 3. Retry a flaky coroutine
    @async_retry(max_attempts=3, delay=0.5, backoff=2.0)
    async def call_api():
        ...

    # 4. Reusable pool
    pool = AsyncPool(max_workers=4)
    async with pool:
        task1 = await pool.submit(fetch, "https://a.com")
        task2 = await pool.submit(fetch, "https://b.com")

    # 5. Race – first successful wins
    fastest = await race(fetch(mirror_a), fetch(mirror_b))

    # 6. Timeout decorator
    @async_timeout(5.0)
    async def slow_operation():
        ...

    # 7. Async cache
    cache = AsyncCache(ttl=60)
    val = await cache.get_or_create("key", expensive_coroutine_factory)
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

__all__ = [
    "gather_with_limit",
    "async_map",
    "async_filter",
    "async_retry",
    "AsyncPool",
    "race",
    "async_timeout",
    "AsyncCache",
]

T = TypeVar("T")
R = TypeVar("R")


async def gather_with_limit(
    coros: Union[Iterable[Coroutine[Any, Any, T]], Iterable[Awaitable[T]]],
    limit: int = 10,
    return_exceptions: bool = False,
) -> List[Union[T, Exception]]:
    """Run an iterable of coroutines / awaitables with a concurrency cap.

    Unlike :func:`asyncio.gather`, which launches every coroutine at once,
    this helper ensures that **at most** ``limit`` coroutines are running
    simultaneously, making it ideal for rate-limited APIs, LLM calls, or
    I/O-bound workloads where unbounded concurrency causes throttling or
    connection errors.

    Parameters
    ----------
    coros
        An iterable of coroutines or awaitables to execute.
    limit
        Maximum number of concurrently in-flight coroutines.  Must be >= 1.
    return_exceptions
        If ``True``, exceptions are returned in the result list instead of
        being raised (mirrors :func:`asyncio.gather` semantics).

    Returns
    -------
    list
        Results in the **same order** as the input iterable.

    Raises
    ------
    ValueError
        If *limit* is less than 1.

    Examples
    --------
    >>> import asyncio
    >>> async def double(x):
    ...     await asyncio.sleep(0.01)
    ...     return x * 2
    >>> # doctest: +SKIP
    >>> asyncio.run(gather_with_limit(double(i) for i in range(5), limit=2))
    [0, 2, 4, 6, 8]
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")

    semaphore = asyncio.Semaphore(limit)
    results: List[Union[T, Exception]] = []
    coro_list = list(coros)

    async def _wrapped(coro: Awaitable[T], index: int) -> None:
        async with semaphore:
            try:
                result = await coro
                results.append((index, result))
            except Exception as exc:
                if return_exceptions:
                    results.append((index, exc))
                else:
                    raise

    tasks = [
        asyncio.ensure_future(_wrapped(c, i)) for i, c in enumerate(coro_list)
    ]
    try:
        await asyncio.gather(*tasks)
    except Exception:
        for t in tasks:
            t.cancel()
        raise

    results.sort(key=lambda pair: pair[0])
    return [r for _, r in results]


async def async_map(
    func: Callable[..., Awaitable[R]],
    iterable: Iterable[T],
    limit: int = 10,
) -> List[R]:
    """Apply an async function to every item, with bounded concurrency.

    This is the async equivalent of :func:`map` but respects a concurrency
    ``limit`` so you do not overwhelm external services.

    Parameters
    ----------
    func
        An async callable that accepts a single item from *iterable*.
    iterable
        Items to process.
    limit
        Maximum concurrent invocations of *func*.

    Returns
    -------
    list
        Results in the same order as *iterable*.

    Examples
    --------
    >>> import asyncio
    >>> async def square(x):
    ...     await asyncio.sleep(0.01)
    ...     return x ** 2
    >>> # doctest: +SKIP
    >>> asyncio.run(async_map(square, [1, 2, 3, 4], limit=2))
    [1, 4, 9, 16]
    """
    items = list(iterable)
    coros = [func(item) for item in items]
    return await gather_with_limit(coros, limit=limit)


async def async_filter(
    func: Callable[[T], Awaitable[bool]],
    iterable: Iterable[T],
    limit: int = 10,
) -> List[T]:
    """Filter an iterable using an async predicate function.

    Items for which *func* returns a truthy value are kept, in original order.

    Parameters
    ----------
    func
        An async callable returning a truthy / falsy value.
    iterable
        Items to filter.
    limit
        Maximum concurrent invocations of *func*.

    Returns
    -------
    list
        Items that passed the predicate, in original order.

    Examples
    --------
    >>> import asyncio
    >>> async def is_even(x):
    ...     await asyncio.sleep(0.01)
    ...     return x % 2 == 0
    >>> # doctest: +SKIP
    >>> asyncio.run(async_filter(is_even, [1, 2, 3, 4, 5, 6]))
    [2, 4, 6]
    """
    items = list(iterable)
    flags = await gather_with_limit(
        [func(item) for item in items], limit=limit
    )
    return [item for item, keep in zip(items, flags) if keep]


def async_retry(
    max_attempts: int = 3,
    delay: float = 0.0,
    backoff: float = 1.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[R]]]:
    """Decorator that retries an async function on failure.

    Parameters
    ----------
    max_attempts
        Maximum number of total attempts (including the first call).
    delay
        Initial delay (seconds) between attempts.
    backoff
        Multiplier applied to *delay* after each failure (``delay *= backoff``).
        Use ``1.0`` for constant delay, ``2.0`` for exponential backoff.
    exceptions
        Tuple of exception types that should trigger a retry.  Other
        exceptions propagate immediately.

    Returns
    -------
    callable
        A decorator that wraps the target async function.

    Examples
    --------
    >>> import asyncio
    >>> call_count = 0
    >>> @async_retry(max_attempts=3, delay=0.01, backoff=2.0)
    ... async def flaky():
    ...     global call_count
    ...     call_count += 1
    ...     if call_count < 3:
    ...         raise ValueError("not yet")
    ...     return "success"
    >>> # doctest: +SKIP
    >>> asyncio.run(flaky())
    'success'
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if backoff < 0:
        raise ValueError("backoff must be >= 0")

    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> R:
            current_delay = delay
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts and current_delay > 0:
                        await asyncio.sleep(current_delay)
                    current_delay *= backoff
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


class AsyncPool:
    """A reusable, bounded-concurrency async worker pool.

    Submit coroutines to the pool and they are executed with at most
    ``max_workers`` concurrently.  The pool manages its own semaphore and
    can be reused across multiple batches without re-creation.

    Parameters
    ----------
    max_workers
        Maximum number of concurrently running coroutines.

    Examples
    --------
    >>> import asyncio
    >>> async def task(n):
    ...     await asyncio.sleep(0.01)
    ...     return n * 10
    >>> # doctest: +SKIP
    >>> async def main():
    ...     pool = AsyncPool(max_workers=3)
    ...     async with pool:
    ...         t1 = await pool.submit(task, 1)
    ...         t2 = await pool.submit(task, 2)
    ...         print(await t1, await t2)
    >>> asyncio.run(main())
    10 20
    """

    def __init__(self, max_workers: int = 5) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = max_workers
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._closed = False

    async def __aenter__(self) -> "AsyncPool":
        self._semaphore = asyncio.Semaphore(self.max_workers)
        self._closed = False
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._semaphore is None or self._closed:
            raise RuntimeError(
                "AsyncPool must be used as an async context manager: "
                "'async with pool:'"
            )

    async def submit(
        self,
        func: Callable[..., Awaitable[R]],
        *args: Any,
        **kwargs: Any,
    ) -> asyncio.Future:
        """Submit a coroutine to the pool.

        Returns a :class:`asyncio.Future` that resolves to the result.
        The coroutine starts executing as soon as a worker slot is available.

        Parameters
        ----------
        func
            An async callable.
        *args, **kwargs
            Positional and keyword arguments forwarded to *func*.

        Returns
        -------
        asyncio.Future
            A future resolving to the coroutine's return value.
        """
        self._ensure_open()
        assert self._semaphore is not None

        async def _run() -> R:
            async with self._semaphore:
                return await func(*args, **kwargs)

        return asyncio.ensure_future(_run())

    async def map(
        self,
        func: Callable[..., Awaitable[R]],
        iterable: Iterable[T],
    ) -> List[R]:
        """Submit *func* over *iterable* and return results in order.

        Convenience method equivalent to :func:`async_map` but using the
        pool's own concurrency limit.
        """
        self._ensure_open()
        items = list(iterable)
        futures = [await self.submit(func, item) for item in items]
        return await asyncio.gather(*futures)


async def race(
    *coros: Coroutine[Any, Any, T],
    timeout: Optional[float] = None,
) -> T:
    """Run multiple coroutines concurrently and return the **first successful** result.

    All other coroutines are cancelled once a winner is determined.  If every
    coroutine raises an exception, the last exception is re-raised.

    Parameters
    ----------
    *coros
        Two or more coroutines to race against each other.
    timeout
        Optional overall timeout in seconds.  If exceeded,
        :class:`asyncio.TimeoutError` is raised.

    Returns
    -------
    T
        The result of the first coroutine to complete successfully.

    Raises
    ------
    ValueError
        If fewer than one coroutine is provided.
    Exception
        The last exception if all coroutines fail.
    asyncio.TimeoutError
        If *timeout* is exceeded.

    Examples
    --------
    >>> import asyncio
    >>> async def fast():
    ...     await asyncio.sleep(0.01)
    ...     return "fast wins"
    >>> async def slow():
    ...     await asyncio.sleep(1.0)
    ...     return "slow wins"
    >>> # doctest: +SKIP
    >>> asyncio.run(race(fast(), slow()))
    'fast wins'
    """
    if not coros:
        raise ValueError("race() requires at least one coroutine")

    tasks = [asyncio.ensure_future(c) for c in coros]
    last_exc: Optional[Exception] = None

    async def _runner() -> T:
        nonlocal last_exc
        pending: Set[asyncio.Task] = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                exc = task.exception()
                if exc is None:
                    for p in pending:
                        p.cancel()
                    return task.result()
                else:
                    last_exc = exc
        assert last_exc is not None
        raise last_exc

    if timeout is not None:
        return await asyncio.wait_for(_runner(), timeout=timeout)
    return await _runner()


def async_timeout(
    seconds: float,
) -> Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[R]]]:
    """Decorator that enforces a timeout on an async function.

    If the wrapped coroutine does not complete within *seconds*,
    :class:`asyncio.TimeoutError` is raised.

    Parameters
    ----------
    seconds
        Maximum allowed execution time in seconds.

    Returns
    -------
    callable
        A decorator wrapping the target async function.

    Examples
    --------
    >>> import asyncio
    >>> @async_timeout(0.05)
    ... async def quick():
    ...     await asyncio.sleep(0.01)
    ...     return "done"
    >>> # doctest: +SKIP
    >>> asyncio.run(quick())
    'done'
    """
    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> R:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


class AsyncCache(Generic[T]):
    """A simple async-safe cache with optional TTL (time-to-live).

    Prevents duplicate concurrent computations: if two callers request the
    same key simultaneously, only one coroutine runs and both receive the
    same result.

    Parameters
    ----------
    ttl
        Time-to-live in seconds.  ``None`` means entries never expire.
    max_size
        Maximum number of entries.  When exceeded, the oldest entry is
        evicted (FIFO).  ``None`` means unlimited.

    Examples
    --------
    >>> import asyncio
    >>> cache = AsyncCache(ttl=60)
    >>> call_count = 0
    >>> async def expensive(x):
    ...     global call_count
    ...     call_count += 1
    ...     return x * 100
    >>> # doctest: +SKIP
    >>> async def main():
    ...     a = await cache.get_or_create("k", lambda: expensive(5))
    ...     b = await cache.get_or_create("k", lambda: expensive(5))
    ...     print(a, b, call_count)  # 500 500 1  - only computed once
    >>> asyncio.run(main())
    """

    def __init__(
        self,
        ttl: Optional[float] = None,
        max_size: Optional[int] = None,
    ) -> None:
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be >= 1 or None")
        self.ttl = ttl
        self.max_size = max_size
        self._store: Dict[Hashable, Tuple[Any, float]] = {}
        self._inflight: Dict[Hashable, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        key: Hashable,
        factory: Callable[[], Awaitable[T]],
    ) -> T:
        """Return the cached value for *key* or compute it via *factory*.

        If a computation for *key* is already in-flight, the caller waits
        for that computation instead of starting a duplicate.

        Parameters
        ----------
        key
            A hashable cache key.
        factory
            A zero-argument async callable that produces the value.

        Returns
        -------
        T
            The cached or freshly computed value.
        """
        now = time.monotonic()
        entry = self._store.get(key)
        if entry is not None:
            value, expires_at = entry
            if expires_at is None or now < expires_at:
                return value
            else:
                del self._store[key]

        if key in self._inflight:
            return await self._inflight[key]

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._inflight[key] = future
        try:
            value = await factory()
            now = time.monotonic()
            expires_at = now + self.ttl if self.ttl is not None else None
            self._store[key] = (value, expires_at)

            if self.max_size is not None and len(self._store) > self.max_size:
                oldest_key = next(iter(self._store))
                if oldest_key != key:
                    del self._store[oldest_key]

            future.set_result(value)
            return value
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            del self._inflight[key]

    def invalidate(self, key: Hashable) -> None:
        """Remove a single key from the cache (if present)."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return False
        _, expires_at = entry
        if expires_at is not None and time.monotonic() >= expires_at:
            return False
        return True
