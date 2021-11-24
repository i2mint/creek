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
    """An iterable type that can actually be used in singledispatch

    >>> assert isinstance([1, 2, 3], IterableType)
    >>> assert not isinstance(2, IterableType)
    """

    def __iter__(self) -> Iterable[IteratorItem]:
        pass


@runtime_checkable
class IteratorType(Protocol):
    """An iterator type that can actually be used in singledispatch

    >>> assert isinstance(iter([1, 2, 3]), IteratorType)
    >>> assert not isinstance([1, 2, 3], IteratorType)
    """

    def __next__(self) -> IteratorItem:
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
no_default = type('no_default', (), {})()


class IteratorExit(BaseException):
    """Raised when an iterator should quit being iterated on, signaling this event
    any process that cares to catch the signal.
    We chose to inherit directly from `BaseException` instead of `Exception`
    for the same reason that `GeneratorExit` does: Because it's not technically
    an error.

    See: https://docs.python.org/3/library/exceptions.html#GeneratorExit
    """


DFLT_INTERRUPT_EXCEPTIONS = (StopIteration, IteratorExit, KeyboardInterrupt)


def iterate_until_exception(iterator, interrupt_exceptions=DFLT_INTERRUPT_EXCEPTIONS):
    while True:
        try:
            next(iterator)
        except interrupt_exceptions:
            print('ending')
            break


def iterable_to_iterator(iterable: Iterable, sentinel=no_sentinel) -> Iterator:
    """Get an iterator from an iterable

    >>> iterable = [1, 2, 3]
    >>> iterator = iterable_to_iterator(iterable)
    >>> assert isinstance(iterator, Iterator)
    >>> assert list(iterator) == iterable

    You can also specify a sentinel, which will result in the iterator stoping just
    before it encounters that sentinel value

    >>> iterable = [1, 2, 3, 4, None, None, 7]
    >>> iterator = iterable_to_iterator(iterable, None)
    >>> assert isinstance(iterator, Iterator)
    >>> list(iterator)
    [1, 2, 3, 4]
    """
    if sentinel is no_sentinel:
        return iter(iterable)
    else:
        return iter(iter(iterable).__next__, sentinel)


def iterator_to_cursor(iterator: Iterator, default=no_default) -> CursorFunc:
    """Get a cursor function for the input iterator.

    >>> iterator = iter([1, 2, 3])
    >>> cursor = iterator_to_cursor(iterator)
    >>> assert callable(cursor)
    >>> assert cursor() == 1
    >>> assert list(cursor_to_iterator(cursor)) == [2, 3]

    Note how we consumed the cursor till the end; by using cursor_to_iterator.
    Indeed, `list(iter(cursor))` wouldn't have worked since a cursor isn't a iterator,
    but a callable to get the items an the iterator would give you.

    You can specify a default. The default has the same role that it has for the
    `next` function: It makes the cursor function return that default when the iterator
    has been "consumed" (i.e. would raise a `StopIteration`).

    >>> iterator = iter([1, 2])
    >>> cursor = iterator_to_cursor(iterator, None)
    >>> assert callable(cursor)
    >>> cursor()
    1
    >>> cursor()
    2

    And then...

    >>> assert cursor() is None
    >>> assert cursor() is None

    forever.

    """
    if default is no_default:
        return partial(next, iterator)
    else:
        return partial(next, iterator, default)


def cursor_to_iterator(cursor: CursorFunc, sentinel=no_sentinel) -> Iterator:
    """Get an iterator from a cursor function.

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
def to_iterator(x: IteratorType, sentinel=no_sentinel):
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
        return x
    else:
        cursor = x.__next__
        return iter(cursor, sentinel)


@to_iterator.register
def _(x: IterableType, sentinel=no_sentinel):
    return to_iterator.__wrapped__(iter(x), sentinel)
    # TODO: Use of __wrapped__ seems hacky. Better way?
    # TODO: Why does to_iterator(iter(x), sentinel) lead to infinite recursion?


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
class Pipe:
    """Simple function composition. That is, gives you a callable that implements input -> f_1 -> ... -> f_n -> output.

    >>> def foo(a, b=2):
    ...     return a + b
    >>> f = Pipe(foo, lambda x: print(f"x: {x}"))
    >>> f(3)
    x: 5

    Notes:
        - Pipe instances don't have a __name__ etc. So some expectations of normal functions are not met.
        - Pipe instance are pickalable (as long as the functions that compose them are)
    """

    def __init__(self, *funcs):

        n_funcs = len(funcs)
        other_funcs = ()
        if n_funcs == 0:
            raise ValueError('You need to specify at least one function!')
        elif n_funcs == 1:
            first_func = last_func = funcs[0]
        else:
            first_func, *other_funcs, last_func = funcs

        self.first_func = first_func
        self.other_funcs = tuple(other_funcs) + (last_func,)

    def __call__(self, *args, **kwargs):
        out = self.first_func(*args, **kwargs)
        for func in self.other_funcs:
            out = func(out)
        return out
