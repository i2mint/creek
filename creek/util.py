"""Utils for creek"""

from functools import (
    WRAPPER_ASSIGNMENTS,
    partial,
    update_wrapper as _update_wrapper,
    wraps as _wraps,
)
from itertools import islice

from typing import Any, Iterable, Iterator, Union
from functools import singledispatch, partial

from typing import Protocol, runtime_checkable

IteratorItem = Any


@runtime_checkable
class IterableType(Protocol):
    def __iter__(self) -> Iterable[IteratorItem]:
        pass


@runtime_checkable
class CursorFunc(Protocol):
    """An argument-less function returning an iterator's values"""

    def __call__(self) -> IteratorItem:
        """Get the next iterator's item and increment the cursor"""


wrapper_assignments = (*WRAPPER_ASSIGNMENTS, '__defaults__', '__kwdefaults__')
update_wrapper = partial(_update_wrapper, assigned=wrapper_assignments)
wraps = partial(_wraps, assigned=wrapper_assignments)


# ---------------------------------------------------------------------------------------
# iteratable, iterator, cursors
no_sentinel = type('no_sentinel', (), {})()


def iterable_to_iterator(iterable: Iterable) -> Iterator:
    """Get an iterator from an iterable

    >>> iterable = [1, 2, 3]
    >>> iterator = iterable_to_iterator(iterable)
    >>> assert isinstance(iterator, Iterator)
    >>> assert list(iterator) == iterable
    """
    return iter(iterable)


def iterator_to_cursor(iterator: Iterator) -> CursorFunc:
    """Get a cursor function for the input iterator.

    >>> iterator = iter([1, 2, 3])
    >>> cursor = iterator_to_cursor(iterator)
    >>> assert callable(cursor)
    >>> assert cursor() == 1
    >>> assert list(cursor_to_iterator(cursor)) == [2, 3]

    Note how we consumed the cursor till the end; by using cursor_to_iterator.
    Indeed, `list(iter(cursor))` wouldn't have worked since a cursor isn't a iterator,
    but a callable to get the items an the iterator would give you.
    """
    return partial(next, iterator)


def cursor_to_iterator(cursor: CursorFunc, sentinel=no_sentinel) -> Iterator:
    """Get an iterator ferom a cursor function.

    A cursor function is a callable that you call (without arguments) to get items of
    data one by one.

    Sometimes, especially in live io contexts, that's the kind interface you're given
    to consume a stream.

    >>> cursor = iter([1, 2, 3]).__next__
    >>> assert not isinstance(cursor, Iterator)
    >>> assert not isinstance(cursor, Iterable)
    >>> assert callable(cursor)

    If you want to consume your stream as an iterator instead, use `cursor_to_iterator`.

    >>> iterator = cursor_to_iterator(cursor)
    >>> assert isinstance(iterator, Iterator)
    >>> list(iterator)
    [1, 2, 3]

    If you want your iterator to stop (without a fuss) as soon as the cursor returns a
    particular element (called a sentinel), say it:

    >>> cursor = iter([1, 2, None, None, 3]).__next__
    >>> iterator = cursor_to_iterator(cursor, sentinel=None)
    >>> list(iterator)
    [1, 2]

    """
    return iter(cursor, sentinel)


def iterable_to_cursor(iterable: Iterable) -> CursorFunc:
    """Get a cursor function from an iterable.
    """
    iterator = iterable_to_iterator(iterable)
    return iterator_to_cursor(iterator)


@singledispatch
def to_iterator(x: IterableType, sentinel=no_sentinel):
    """Get an iterator from an iterable or a cursor function

    >>> from typing import Iterator
    >>> it = to_iterator([1, 2, 3])
    >>> assert isinstance(it, Iterator)
    >>> list(it)
    [1, 2, 3]
    >>> list(it)
    []

    >>> cursor = iter([1, 2, 3]).__next__
    >>> assert isinstance(cursor, CursorFunc)
    >>> it = to_iterator(cursor)
    >>> assert isinstance(it, Iterator)
    >>> list(it)
    [1, 2, 3]
    >>> list(it)
    []

    You can use sentinels too

    >>> list(to_iterator([1, 2, None, 4], sentinel=None))
    [1, 2]
    >>> cursor = iter([1, 2, 3, 4, 5]).__next__
    >>> list(to_iterator(cursor, sentinel=4))
    [1, 2, 3]
    """
    if sentinel is no_sentinel:
        return iter(x)
    else:
        cursor = iter(x).__next__
        return to_iterator(cursor, sentinel)


@to_iterator.register
def _(x: CursorFunc, sentinel=no_sentinel):
    return iter(x, sentinel)


# ---------------------------------------------------------------------------------------

no_such_item = type('NoSuchItem', (), {})()


class stream_util:
    def always_true(*args, **kwargs):
        return True

    def do_nothing(*args, **kwargs):
        pass

    def rewind(self, instance):
        instance.seek(0)

    def skip_lines(self, instance, n_lines_to_skip=0):
        instance.seek(0)


class PreIter:
    def skip_items(self, instance, n):
        return islice(instance, n, None)


def cls_wrap(cls, obj):
    if isinstance(obj, type):

        @wraps(obj, updated=())
        class Wrap(cls):
            @wraps(obj.__init__)
            def __init__(self, *args, **kwargs):
                wrapped = obj(*args, **kwargs)
                super().__init__(wrapped)

        # Wrap.__signature__ = signature(obj)

        return Wrap
    else:
        return cls(obj)


# TODO: Make identity_func "identifiable". If we use the following one, we can use == to detect it's use,
# TODO: ... but there may be a way to annotate, register, or type any identity function so it can be detected.


def identity_func(x):
    return x


static_identity_method = staticmethod(identity_func)


# TODO: Understand why this doesn't work:
# --> TypeError: Invalid annotation for 'iterable'. typing.AsyncIterable is not a class.
# @singledispatch
# def iterable_to_iterator(iterable: Iterable) -> Iterator:
#     return iter(iterable)
# @iterable_to_iterator.register
# def _(iterable: AsyncIterable) -> AsyncIterator:
#     return aiter(iterable)
