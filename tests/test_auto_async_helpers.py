"""
Comprehensive pytest suite for ilovetools.utils.async_helpers

Covers every public function and class, including edge cases, exceptions,
concurrency limits, TTL expiry, retry logic, and timeout enforcement.
"""

import asyncio
import pytest

from ilovetools.utils.async_helpers import (
    gather_with_limit,
    async_map,
    async_filter,
    async_retry,
    AsyncPool,
    race,
    async_timeout,
    AsyncCache,
)


async def _double(x):
    await asyncio.sleep(0.001)
    return x * 2


class TestGatherWithLimit:
    @pytest.mark.asyncio
    async def test_basic_results_in_order(self):
        results = await gather_with_limit((_double(i) for i in range(5)), limit=2)
        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_limit_one_sequential(self):
        results = await gather_with_limit((_double(i) for i in range(3)), limit=1)
        assert results == [0, 2, 4]

    @pytest.mark.asyncio
    async def test_empty_iterable(self):
        results = await gather_with_limit([], limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_return_exceptions_true(self):
        async def boom(x):
            raise ValueError(f"boom {x}")

        results = await gather_with_limit(
            [boom(1), _double(2)], limit=2, return_exceptions=True
        )
        assert isinstance(results[0], ValueError)
        assert results[1] == 4

    @pytest.mark.asyncio
    async def test_exception_propagates_when_not_returning(self):
        async def boom(x):
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            await gather_with_limit([boom(1), _double(2)], limit=2)

    @pytest.mark.asyncio
    async def test_invalid_limit_raises(self):
        with pytest.raises(ValueError, match="limit must be >= 1"):
            await gather_with_limit([], limit=0)

    @pytest.mark.asyncio
    async def test_concurrency_actually_bounded(self):
        current = 0
        peak = 0

        async def tracker(x):
            nonlocal current, peak
            current += 1
            peak = max(peak, current)
            await asyncio.sleep(0.02)
            current -= 1
            return x

        await gather_with_limit((tracker(i) for i in range(20)), limit=3)
        assert peak <= 3


class TestAsyncMap:
    @pytest.mark.asyncio
    async def test_basic_map(self):
        results = await async_map(_double, [1, 2, 3, 4], limit=2)
        assert results == [2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_empty_input(self):
        results = await async_map(_double, [], limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        async def slow_double(x):
            await asyncio.sleep(0.05 - x * 0.01)
            return x * 2

        results = await async_map(slow_double, [0, 1, 2, 3, 4], limit=5)
        assert results == [0, 2, 4, 6, 8]


class TestAsyncFilter:
    @pytest.mark.asyncio
    async def test_basic_filter(self):
        async def is_even(x):
            await asyncio.sleep(0.001)
            return x % 2 == 0

        result = await async_filter(is_even, [1, 2, 3, 4, 5, 6], limit=3)
        assert result == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_all_pass(self):
        async def always_true(x):
            return True

        result = await async_filter(always_true, [1, 2, 3])
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_none_pass(self):
        async def always_false(x):
            return False

        result = await async_filter(always_false, [1, 2, 3])
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_input(self):
        async def pred(x):
            return True

        result = await async_filter(pred, [])
        assert result == []


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self):
        call_count = 0

        @async_retry(max_attempts=3, delay=0.001)
        async def ok():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await ok()
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        call_count = 0

        @async_retry(max_attempts=3, delay=0.001, backoff=2.0)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "success"

        result = await flaky()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_attempts_fail(self):
        call_count = 0

        @async_retry(max_attempts=2, delay=0.001)
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("always")

        with pytest.raises(ValueError, match="always"):
            await always_fail()
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_only_specified_exceptions_retry(self):
        call_count = 0

        @async_retry(max_attempts=3, delay=0.001, exceptions=(ValueError,))
        async def mixed():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retried")

        with pytest.raises(TypeError, match="not retried"):
            await mixed()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_max_attempts(self):
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            async_retry(max_attempts=0)

    @pytest.mark.asyncio
    async def test_zero_delay_works(self):
        call_count = 0

        @async_retry(max_attempts=3, delay=0.0)
        async def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("retry")
            return "done"

        result = await fail_twice()
        assert result == "done"
        assert call_count == 3


class TestAsyncPool:
    @pytest.mark.asyncio
    async def test_basic_submit(self):
        async with AsyncPool(max_workers=2) as pool:
            f1 = await pool.submit(_double, 5)
            f2 = await pool.submit(_double, 10)
            assert await f1 == 10
            assert await f2 == 20

    @pytest.mark.asyncio
    async def test_map(self):
        async with AsyncPool(max_workers=3) as pool:
            results = await pool.map(_double, [1, 2, 3, 4, 5])
            assert results == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_concurrency_bounded(self):
        current = 0
        peak = 0

        async def tracker(x):
            nonlocal current, peak
            current += 1
            peak = max(peak, current)
            await asyncio.sleep(0.02)
            current -= 1
            return x

        async with AsyncPool(max_workers=2) as pool:
            futures = [await pool.submit(tracker, i) for i in range(10)]
            await asyncio.gather(*futures)

        assert peak <= 2

    @pytest.mark.asyncio
    async def test_submit_without_context_raises(self):
        pool = AsyncPool(max_workers=2)
        with pytest.raises(RuntimeError, match="async context manager"):
            await pool.submit(_double, 1)

    @pytest.mark.asyncio
    async def test_invalid_max_workers(self):
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            AsyncPool(max_workers=0)


class TestRace:
    @pytest.mark.asyncio
    async def test_fastest_wins(self):
        async def fast():
            await asyncio.sleep(0.01)
            return "fast"

        async def slow():
            await asyncio.sleep(1.0)
            return "slow"

        result = await race(fast(), slow())
        assert result == "fast"

    @pytest.mark.asyncio
    async def test_first_fails_second_wins(self):
        async def fails():
            await asyncio.sleep(0.01)
            raise ValueError("fail")

        async def succeeds():
            await asyncio.sleep(0.02)
            return "winner"

        result = await race(fails(), succeeds())
        assert result == "winner"

    @pytest.mark.asyncio
    async def test_all_fail_raises_last(self):
        async def fail_a():
            await asyncio.sleep(0.01)
            raise ValueError("a")

        async def fail_b():
            await asyncio.sleep(0.02)
            raise ValueError("b")

        with pytest.raises(ValueError, match="b"):
            await race(fail_a(), fail_b())

    @pytest.mark.asyncio
    async def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one coroutine"):
            await race()

    @pytest.mark.asyncio
    async def test_timeout(self):
        async def slow():
            await asyncio.sleep(1.0)
            return "slow"

        with pytest.raises(asyncio.TimeoutError):
            await race(slow(), timeout=0.05)


class TestAsyncTimeout:
    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        @async_timeout(1.0)
        async def quick():
            await asyncio.sleep(0.01)
            return "done"

        assert await quick() == "done"

    @pytest.mark.asyncio
    async def test_exceeds_timeout(self):
        @async_timeout(0.02)
        async def slow():
            await asyncio.sleep(1.0)
            return "never"

        with pytest.raises(asyncio.TimeoutError):
            await slow()

    @pytest.mark.asyncio
    async def test_invalid_seconds(self):
        with pytest.raises(ValueError, match="seconds must be > 0"):
            async_timeout(0)

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        @async_timeout(1.0)
        async def my_func():
            return "ok"

        assert my_func.__name__ == "my_func"


class TestAsyncCache:
    @pytest.mark.asyncio
    async def test_basic_get_or_create(self):
        cache = AsyncCache(ttl=60)
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return 42

        assert await cache.get_or_create("k", factory) == 42
        assert await cache.get_or_create("k", factory) == 42
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_keys(self):
        cache = AsyncCache()
        assert await cache.get_or_create("a", lambda: _double(1)) == 2
        assert await cache.get_or_create("b", lambda: _double(2)) == 4

    @pytest.mark.asyncio
    async def test_ttl_expiry(self):
        cache = AsyncCache(ttl=0.05)
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return "val"

        await cache.get_or_create("k", factory)
        await asyncio.sleep(0.06)
        await cache.get_or_create("k", factory)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_ttl_never_expires(self):
        cache = AsyncCache(ttl=None)
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return "val"

        await cache.get_or_create("k", factory)
        await asyncio.sleep(0.02)
        await cache.get_or_create("k", factory)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_inflight_deduplication(self):
        cache = AsyncCache(ttl=60)
        call_count = 0

        async def slow_factory():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return "result"

        r1, r2 = await asyncio.gather(
            cache.get_or_create("k", slow_factory),
            cache.get_or_create("k", slow_factory),
        )
        assert r1 == r2 == "result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_invalidate(self):
        cache = AsyncCache(ttl=60)
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return "val"

        await cache.get_or_create("k", factory)
        cache.invalidate("k")
        await cache.get_or_create("k", factory)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_clear(self):
        cache = AsyncCache(ttl=60)
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return "val"

        await cache.get_or_create("k", factory)
        cache.clear()
        assert len(cache) == 0
        await cache.get_or_create("k", factory)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_size_eviction(self):
        cache = AsyncCache(ttl=None, max_size=2)
        call_count = 0

        async def factory(x):
            nonlocal call_count
            call_count += 1
            return x

        await cache.get_or_create("a", lambda: factory(1))
        await cache.get_or_create("b", lambda: factory(2))
        await cache.get_or_create("c", lambda: factory(3))
        assert len(cache) <= 2

    @pytest.mark.asyncio
    async def test_contains(self):
        cache = AsyncCache(ttl=60)

        async def factory():
            return "val"

        assert "k" not in cache
        await cache.get_or_create("k", factory)
        assert "k" in cache

    @pytest.mark.asyncio
    async def test_factory_exception_propagates(self):
        cache = AsyncCache(ttl=60)

        async def boom():
            raise ValueError("factory failed")

        with pytest.raises(ValueError, match="factory failed"):
            await cache.get_or_create("k", boom)

    @pytest.mark.asyncio
    async def test_invalid_max_size(self):
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            AsyncCache(max_size=0)
