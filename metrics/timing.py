"""Timing utilities for lightweight latency tracking."""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Callable, Optional, TypeVar

import pro_metrics

F = TypeVar("F", bound=Callable[..., Any])


def timed(
    func: Optional[F] = None, *, name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator to record the execution time of *func*.

    The *name* parameter allows overriding the reported metric label. The
    decorator works with both synchronous and asynchronous callables.
    """

    def decorator(f: F) -> F:
        metric_name = name or f.__qualname__
        if asyncio.iscoroutinefunction(f):
            @functools.wraps(f)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return await f(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    pro_metrics.record_latency(metric_name, duration)

            return async_wrapper  # type: ignore[misc]
        else:
            @functools.wraps(f)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return f(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    pro_metrics.record_latency(metric_name, duration)

            return sync_wrapper  # type: ignore[misc]

    if func is None:
        return decorator
    return decorator(func)
