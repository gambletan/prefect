"""
Thread-safe utilities for working with asynchronous event loops.
"""

import asyncio
import concurrent.futures
import contextvars
import functools
from collections.abc import Coroutine
from typing import Any, Callable, Optional, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def get_running_loop() -> Optional[asyncio.AbstractEventLoop]:
    """
    Get the current running loop.

    Returns `None` if there is no running loop.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def as_asyncio_future(
    future: concurrent.futures.Future[T],
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> asyncio.Future[T]:
    """
    Bridge a concurrent future into an asyncio future without propagating the
    source thread's contextvars into the destination loop.
    """
    loop = loop or get_running_loop()
    if loop is None:
        raise RuntimeError("An event loop is required to bridge this future.")

    destination: asyncio.Future[T] = loop.create_future()

    def transfer_result(source: concurrent.futures.Future[T]) -> None:
        if loop.is_closed():
            return

        def set_result() -> None:
            if destination.done():
                return

            if source.cancelled():
                destination.cancel()
                return

            try:
                destination.set_result(source.result())
            except BaseException as exc:
                destination.set_exception(exc)

        loop.call_soon_threadsafe(set_result, context=contextvars.Context())

    def cancel_source(source: asyncio.Future[T]) -> None:
        if source.cancelled() and not future.done():
            future.cancel()

    future.add_done_callback(transfer_result)
    destination.add_done_callback(cancel_source)
    return destination


def call_soon_in_loop(
    __loop: asyncio.AbstractEventLoop,
    __fn: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> concurrent.futures.Future[T]:
    """
    Run a synchronous call in an event loop's thread from another thread.

    This function is non-blocking and safe to call from an asynchronous context.

    Returns a future that can be used to retrieve the result of the call.
    """
    future: concurrent.futures.Future[T] = concurrent.futures.Future()

    @functools.wraps(__fn)
    def wrapper() -> None:
        try:
            result = __fn(*args, **kwargs)
        except BaseException as exc:
            future.set_exception(exc)
            if not isinstance(exc, Exception):
                raise
        else:
            future.set_result(result)

    # `call_soon...` returns a `Handle` object which doesn't provide access to the
    # result of the call. We wrap the call with a future to facilitate retrieval.
    if __loop is get_running_loop():
        __loop.call_soon(wrapper)
    else:
        __loop.call_soon_threadsafe(wrapper)

    return future


async def run_coroutine_in_loop_from_async(
    __loop: asyncio.AbstractEventLoop, __coro: Coroutine[Any, Any, T]
) -> T:
    """
    Run an asynchronous call in an event loop from an asynchronous context.

    Returns an awaitable that returns the result of the coroutine.
    """
    if __loop is get_running_loop():
        return await __coro
    else:
        return await as_asyncio_future(asyncio.run_coroutine_threadsafe(__coro, __loop))
