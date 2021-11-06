"""Utils to deal with async iteration


Making singledispatch work:

```
from typing import Generator, Iterator, Iterable, AsyncIterable, AsyncIterator
from functools import singledispatch, partial

from typing import Protocol, runtime_checkable

@runtime_checkable
class IterableType(Protocol):
    def __iter__(self):
        pass

@runtime_checkable
class CursorFunc(Protocol):
    def __call__(self):
        pass

@singledispatch
def to_iterator(x: IterableType):
    return iter(x)

will_never_happen = object()

@to_iterator.register
def _(x: CursorFunc):
    return iter(x, will_never_happen)

assert list(to_iterator([1,2,3])) == [1, 2, 3]

f = partial(next, iter([1,2,3]))
assert list(to_iterator(f)) == [1, 2, 3]
```

Trying to make async iterators/iterables/cursor_funcs utils

```
import asyncio


async def ticker(to=3, delay=0.5):
    # Yield numbers from 0 to `to` every `delay` seconds.
    for i in range(to):
        yield i
        await asyncio.sleep(delay)

async def my_aiter(async_iterable):
    async for i in async_iterable:
        yield i

t = [i async for i in my_aiter(ticker(3, 0.2))]
assert t == [0, 1, 2]

# t = list(my_aiter(ticker(3, 0.2)))
# # TypeError: 'async_generator' object is not iterable
# # and
# t = await list(ticker(3, 0.2))
# # TypeError: 'async_generator' object is not iterable

# But...

async def alist(async_iterable):
    return [i async for i in async_iterable]

t = await alist(ticker(3, 0.2))
assert t == [0, 1, 2]
```


"""

from functools import partial
from typing import (
    Callable,
    Any,
    NewType,
    Iterable,
    AsyncIterable,
    Iterator,
    AsyncIterator,
    Union,
)

IterableType = Union[Iterable, AsyncIterable]
IteratorType = Union[Iterator, AsyncIterator]
CursorFunc = NewType('CursorFunc', Callable[[], Any])
CursorFunc.__doc__ = "An argument-less function returning an iterator's values"


# ---------------------------------------------------------------------------------------
# iteratable, iterator, cursors
no_sentinel = type('no_sentinel', (), {})()

try:
    aiter  # exists in python 3.10+
    # Note: doesn't have the sentinel though!!
except NameError:

    async def aiter(iterable: AsyncIterable) -> AsyncIterator:
        if not isinstance(iterable, AsyncIterable):
            raise TypeError(f'aiter expected an AsyncIterable, got {type(iterable)}')
        if isinstance(iterable, AsyncIterator):
            return iterable
        return (i async for i in iterable)


async def aiter_with_sentinel(cursor_func: CursorFunc, sentinel: Any) -> AsyncIterator:
    """Like iter(async_callable, sentinel) builtin but for async callables"""
    while (value := await cursor_func()) is not sentinel:
        yield value


def iterable_to_iterator(iterable: IterableType) -> IteratorType:
    """Get an iterator from an iterable (whether async or not)

    >>> iterable = [1, 2, 3]
    >>> iterator = iterable_to_iterator(iterable)
    >>> assert isinstance(iterator, Iterator)
    >>> assert list(iterator) == iterable
    """
    if isinstance(iterable, AsyncIterable):
        return aiter(iterable)
    return iter(iterable)


def iterator_to_cursor(iterator: Iterator) -> CursorFunc:
    return partial(next, iterator)


def cursor_to_iterator(cursor: CursorFunc, sentinel=no_sentinel) -> Iterator:
    return iter(cursor, no_sentinel)


def iterable_to_cursor(iterable: Iterable, sentinel=no_sentinel) -> CursorFunc:
    iterator = iterable_to_iterator(iterable)
    if sentinel is no_sentinel:
        return iterator_to_cursor(iterator)
    else:
        return partial(next, iterator, sentinel)
