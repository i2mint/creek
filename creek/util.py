"""Utils for creek"""
import itertools
from functools import (
    WRAPPER_ASSIGNMENTS,
    partial,
    update_wrapper as _update_wrapper,
    wraps as _wraps,
)
from itertools import islice

from typing import (
    Callable,
    Any,
    Iterable,
    AsyncIterable,
    Iterator,
    AsyncIterator,
    NewType,
)
from functools import singledispatch, partial

from atypes import Slab, MyType
from i2.multi_object import FuncFanout, ContextFanout, MultiObj

CursorFunc = NewType('CursorFunc', Callable[[], Any])
CursorFunc.__doc__ = "An argument-less function returning an iterator's values"

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


# ---------------------------------------------------------------------------------------


class _MultiIterator(MultiObj):
    """Helper class for DictZip"""

    def _gen_next(self):
        for name, iterator in self.objects.items():
            yield name, next(iterator, None)

    def __next__(self) -> dict:
        return dict(self._gen_next())


class DictZip:
    def __init__(self, *unnamed, takewhile=None, **named):
        self.multi_iterator = _MultiIterator(*unnamed, **named)
        self.objects = self.multi_iterator.objects
        self.takewhile = takewhile

    def __iter__(self):
        while True:
            x = next(self.multi_iterator)
            if not self.takewhile(x):
                break
            yield x


class MultiIterable:
    def __init__(self, *unnamed, **named):
        self.multi_iterator = _MultiIterator(*unnamed, **named)
        self.objects = self.multi_iterator.objects

    def __iter__(self):
        while True:
            yield next(self.multi_iterator)

    def takewhile(self, predicate=None):
        """itertools.takewhile applied to self, with a bit of syntactic sugar
        There's nothing to stop the iteration"""
        if predicate is None:
            predicate = lambda x: True  # always true
        return itertools.takewhile(predicate, self)


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
