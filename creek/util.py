"""Utils for creek"""

from functools import (
    WRAPPER_ASSIGNMENTS,
    partial,
    update_wrapper as _update_wrapper,
    wraps as _wraps,
)
from itertools import islice

from typing import Any, Iterable, Iterator, Union, NewType
from typing import Protocol, runtime_checkable

from functools import singledispatch, partial


def iterize(func, name=None):
    """From an In->Out function, makes a Iterator[In]->Itertor[Out] function.

    >>> f = lambda x: x * 10
    >>> f(2)
    20
    >>> iterized_f = iterize(f)
    >>> list(iterized_f(iter([1,2,3])))
    [10, 20, 30]

    """
    iterized_func = partial(map, func)
    if name is not None:
        iterized_func.__name__ = name
    return iterized_func


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


IterType = NewType('IterType', Union[IteratorType, IterableType, CursorFunc])
IterType.__doc__ = 'A type that can be made into an iterator'

wrapper_assignments = (*WRAPPER_ASSIGNMENTS, '__defaults__', '__kwdefaults__')
update_wrapper = partial(_update_wrapper, assigned=wrapper_assignments)
wraps = partial(_wraps, assigned=wrapper_assignments)


# ---------------------------------------------------------------------------------------
# iteratable, iterator, cursors
# TODO: If bring i2 as dependency, use mk_sentinel here
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


from inspect import signature, Signature


# Note: Pipe code is completely independent (with inspect imports signature & Signature)
#  If you only need simple pipelines, use this, or even copy/paste it where needed.
# TODO: Public interface mis-aligned with i2. funcs list here, in i2 it's dict. Align?
#  If we do so, it would be a breaking change since any dependents that expect funcs
#  to be a list of funcs will iterate over a iterable of names instead.
class Pipe:
    """Simple function composition. That is, gives you a callable that implements input -> f_1 -> ... -> f_n -> output.

    >>> def foo(a, b=2):
    ...     return a + b
    >>> f = Pipe(foo, lambda x: print(f"x: {x}"))
    >>> f(3)
    x: 5
    >>> len(f)
    2

    You can name functions, but this would just be for documentation purposes.
    The names are completely ignored.

    >>> g = Pipe(
    ...     add_numbers = lambda x, y: x + y,
    ...     multiply_by_2 = lambda x: x * 2,
    ...     stringify = str
    ... )
    >>> g(2, 3)
    '10'
    >>> len(g)
    3

    Notes:
        - Pipe instances don't have a __name__ etc. So some expectations of normal functions are not met.
        - Pipe instance are pickalable (as long as the functions that compose them are)

    You can specify a single functions:

    >>> Pipe(lambda x: x + 1)(2)
    3

    but

    >>> Pipe()
    Traceback (most recent call last):
      ...
    ValueError: You need to specify at least one function!

    You can specify an instance name and/or doc with the special (reserved) argument
    names ``__name__`` and ``__doc__`` (which therefore can't be used as function names):

    >>> f = Pipe(map, add_it=sum, __name__='map_and_sum', __doc__='Apply func and add')
    >>> f(lambda x: x * 10, [1, 2, 3])
    60
    >>> f.__name__
    'map_and_sum'
    >>> f.__doc__
    'Apply func and add'

    """

    funcs = ()

    def __init__(self, *funcs, **named_funcs):
        named_funcs = self._process_reserved_names(named_funcs)
        funcs = list(funcs) + list(named_funcs.values())
        self.funcs = funcs
        n_funcs = len(funcs)
        if n_funcs == 0:
            raise ValueError('You need to specify at least one function!')

        elif n_funcs == 1:
            other_funcs = ()
            first_func = last_func = funcs[0]
        else:
            first_func, *other_funcs = funcs
            *_, last_func = other_funcs

        self.__signature__ = Pipe._signature_from_first_and_last_func(
            first_func, last_func
        )
        self.first_func, self.other_funcs = first_func, other_funcs

    _reserved_names = ('__name__', '__doc__')

    def _process_reserved_names(self, named_funcs):
        for name in self._reserved_names:
            if (value := named_funcs.pop(name, None)) is not None:
                setattr(self, name, value)
        return named_funcs

    def __call__(self, *args, **kwargs):
        out = self.first_func(*args, **kwargs)
        for func in self.other_funcs:
            out = func(out)
        return out

    def __len__(self):
        return len(self.funcs)

    _dflt_signature = Signature.from_callable(lambda *args, **kwargs: None)

    @staticmethod
    def _signature_from_first_and_last_func(first_func, last_func):
        try:
            input_params = signature(first_func).parameters.values()
        except ValueError:  # function doesn't have a signature, so take default
            input_params = Pipe._dflt_signature.parameters.values()
        try:
            return_annotation = signature(last_func).return_annotation
        except ValueError:  # function doesn't have a signature, so take default
            return_annotation = Pipe._dflt_signature.return_annotation
        return Signature(tuple(input_params), return_annotation=return_annotation)
