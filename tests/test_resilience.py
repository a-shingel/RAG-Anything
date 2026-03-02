"""Tests for retry and resilience utilities."""

import asyncio
import pytest

from raganything.resilience import retry, async_retry, CircuitBreaker


class TestSyncRetry:
    def test_succeeds_first_try(self):
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def good_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert good_func() == "ok"
        assert call_count == 1

    def test_retries_on_transient_error(self):
        call_count = 0

        @retry(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=[ConnectionError],
        )
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "recovered"

        assert flaky_func() == "recovered"
        assert call_count == 3

    def test_raises_after_max_attempts(self):
        @retry(
            max_attempts=2,
            base_delay=0.01,
            retryable_exceptions=[ConnectionError],
        )
        def always_fail():
            raise ConnectionError("permanent")

        with pytest.raises(ConnectionError, match="permanent"):
            always_fail()

    def test_does_not_retry_non_retryable(self):
        call_count = 0

        @retry(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=[ConnectionError],
        )
        def type_error_func():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            type_error_func()
        assert call_count == 1

    def test_on_retry_callback(self):
        retries_seen = []

        def on_retry_cb(exc, attempt, delay):
            retries_seen.append((type(exc).__name__, attempt))

        call_count = 0

        @retry(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=[OSError],
            on_retry=on_retry_cb,
        )
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("oops")
            return "ok"

        assert flaky() == "ok"
        assert len(retries_seen) == 2
        assert retries_seen[0] == ("OSError", 1)
        assert retries_seen[1] == ("OSError", 2)


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        call_count = 0

        @async_retry(max_attempts=3, base_delay=0.01)
        async def good_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert await good_func() == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        call_count = 0

        @async_retry(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=[ConnectionError],
        )
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "recovered"

        assert await flaky_func() == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        @async_retry(
            max_attempts=2,
            base_delay=0.01,
            retryable_exceptions=[TimeoutError],
        )
        async def always_fail():
            raise TimeoutError("permanent")

        with pytest.raises(TimeoutError, match="permanent"):
            await always_fail()


class TestCircuitBreaker:
    def test_closed_state_allows_calls(self):
        cb = CircuitBreaker(failure_threshold=3, name="test")

        @cb
        def ok_func():
            return "ok"

        assert ok_func() == "ok"
        assert cb.state == "closed"

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1, name="test")

        @cb
        def fail_func():
            raise RuntimeError("fail")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                fail_func()

        assert cb.state == "open"

        with pytest.raises(CircuitBreaker.CircuitBreakerOpen):
            fail_func()

    def test_half_open_after_timeout(self):
        import time

        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05, name="test")

        @cb
        def fail_func():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            fail_func()

        assert cb.state == "open"
        time.sleep(0.06)
        assert cb.state == "half-open"

    def test_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=3, name="test")
        cb._failure_count = 2

        @cb
        def ok_func():
            return "ok"

        ok_func()
        assert cb._failure_count == 0
        assert cb.state == "closed"
