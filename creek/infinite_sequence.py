"""
Objects that support some list-like read operations on an unbounded stream.
Essentially, trying to give you the impression that you have read access to infinite list,
with some (parametrizable) limitations.

"""
# TODO: Build up extensive relations expression and handling, but InfiniteSeq only uses BEFORE (past).
#  Consider simplifying.

from collections import deque
from typing import Iterable, Tuple, Union, Callable, Any, NewType
from functools import wraps, partial, partialmethod
from enum import Enum
from operator import le, lt, ge, gt, itemgetter
from threading import Lock
from itertools import count, islice

Number = Union[int, float]  # TODO: existing builtin to specify real numbers?

opposite_op = {
    le: gt,
    lt: ge,
    ge: lt,
    gt: le,
}


# TODO: Check performance for string versus int enum values
class Relations(Enum):
    """Point-interval and interval-interval relations.

    See Allen's interval algebra for (some of the) interval relations
    (https://en.wikipedia.org/wiki/Allen%27s_interval_algebra).
    """

    # simple relations, that can be used between
    # (X: point, Y: interval) or (X: interval, Y: interval) pairs
    BEFORE = 'Some of X happens BEFORE Y'
    DURING = 'All of X happens within Y'
    AFTER = 'Some of X happens AFTER Y'

    # Extras (Allen's)
    PRECEDES = 'X precedes Y: All of X happens before Y'
    MEETS = 'X meets Y: When X ends, Y starts'
    OVERLAPS = 'X overlaps Y: Point is AFTER interval'
    STARTS = 'X starts at the same time as Y (and finishes no later)'
    FINISHES = 'X finishes Y: Point is AFTER interval'
    EQUAL = 'X is equal to Y'


def validate_interval(interval):
    """Asserts that input is a valid interval, raising a ValueError if not"""
    try:
        bt, tt = interval
        assert bt <= tt
        return bt, tt
    except Exception as e:
        raise ValueError(f'Not a valid interval: {interval}')


# TODO: Validate intervals (assert x[0] <= x[1] and )?
def simple_interval_relationship(
    x: Tuple[Number, Number],
    y: Tuple[Number, Number],
    above_bt: Callable = ge,
    below_tt: Callable = lt,
):
    """Get the simple relationship between intervals x and y.

    :param x: An point (a number) or an interval (a 2-tuple of numbers).
    :param y: An interval; a 2-tuple of numbers.
    :param above_bt: a above_bt(x_bt, y_bt) boolean function (ge or gt) deciding if x starts after y does.
    :param below_tt: a below_tt(x_tt, y_tt) boolean function (lt or le) deciding if x ends before y does.
    :return: One of three relations
        Relations.BEFORE if some of x is below y,
        Relations.AFTER if some of x is after y,
        Relations.DURING if x is entirely with y

    The target ``y`` interval is expressed only by it's bounds, but we don't know if
     these are inclusive or not. The ``below_bt`` and ``above_tt`` allow us to express
     that by expressing how below the lowest (bt) bound and what higher than highest
     (tt) bound are defined.

    The function is meant to be curried (partial), for example:

    >>> from functools import partial
    >>> from operator import le, lt, ge, gt
    >>> default = simple_interval_relationship  # uses below_bt=ge, above_tt=lt
    >>> including_bounds = partial(simple_interval_relationship, above_bt=ge, below_tt=le)
    >>> excluding_bounds = partial(simple_interval_relationship, above_bt=gt, below_tt=lt)

    Take ``(4, 8)`` as the target interval, and want to query the relationship of other
    points and intervals with it.
    No matter what the function is, they will always agree on any intervals that don't
    share any bounds.

    >>> for relation_func in (default, including_bounds, excluding_bounds):
    ...     print (
    ...         relation_func(3, (4, 8)),
    ...         relation_func(5, (4, 8)),
    ...         relation_func(9, (4, 8)),
    ...         relation_func((3, 7), (4, 8)),
    ...         relation_func((5, 7), (4, 8)),
    ...         relation_func((7, 9), (4, 8))
    ... )
    Relations.BEFORE Relations.DURING Relations.AFTER Relations.BEFORE Relations.DURING Relations.AFTER
    Relations.BEFORE Relations.DURING Relations.AFTER Relations.BEFORE Relations.DURING Relations.AFTER
    Relations.BEFORE Relations.DURING Relations.AFTER Relations.BEFORE Relations.DURING Relations.AFTER

    But if the two intervals share some bounds, these functions will diverge.

    >>> for relation_func in (default, including_bounds, excluding_bounds):
    ...     print (
    ...         relation_func(4, (4, 8)),
    ...         relation_func(8, (4, 8)),
    ...         relation_func((4, 7), (4, 8)),
    ...         relation_func((4, 8), (4, 8)),
    ...         relation_func((5, 8), (4, 8))
    ... )
    Relations.DURING Relations.AFTER Relations.DURING Relations.AFTER Relations.AFTER
    Relations.DURING Relations.DURING Relations.DURING Relations.DURING Relations.DURING
    Relations.BEFORE Relations.AFTER Relations.BEFORE Relations.BEFORE Relations.AFTER

    The function can be used with the FIRST argument being a slice object as well.
    This can then be used to enable [i:j] access.

    >>> for relation_func in (default, including_bounds, excluding_bounds):
    ...     print (
    ...         relation_func(slice(4, 7), (4, 8)),
    ...         relation_func(slice(4, 8), (4, 8)),
    ...         relation_func(slice(5, 8), (4, 8))
    ... )
    Relations.DURING Relations.AFTER Relations.AFTER
    Relations.DURING Relations.DURING Relations.DURING
    Relations.BEFORE Relations.BEFORE Relations.AFTER

    """
    if isinstance(x, slice):
        x_bt, x_tt = validate_interval((x.start or 0, x.stop or 0))
    elif isinstance(x, Iterable):
        x_bt, x_tt = validate_interval(x)
    else:
        x_bt, x_tt = x, x  # consider a point to be the (x, x) interval
    y_bt, y_tt = validate_interval(y)
    if not above_bt(x_bt, y_bt):  # meaning x_bt > y_bt (or >=, depending on above_bt)
        return Relations.BEFORE
    elif below_tt(x_tt, y_tt):  # meaning x_bt < y_tt (or <=, depending on below_tt)
        return Relations.DURING
    else:
        return Relations.AFTER


# Error handling #######################################################################################################
class ExceptionRaiserCallbackMixin:
    """Make the instance callable and have the effect of raising the instance.
    Meant to add to an exception class so that instances of this class can be used as callbacks that raise the error"""

    dflt_args = ()

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            if isinstance(self.dflt_args, str):
                self.dflt_args = (self.dflt_args,)
            args = self.dflt_args
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise self


# TODO: Include all 13 of Allen's interval algebra relations? Enum/sentinels and errors for these events?
#  (https://en.wikipedia.org/wiki/Allen%27s_interval_algebra#Relations)
class NotDuringError(ExceptionRaiserCallbackMixin, IndexError):
    """IndexError that indicates that there was an attempt to index some data that is not contained in the buffer
    (i.e. is that a part of the request is NO LONGER, or NOT YET covered by the buffer)"""

    dflt_args = 'Some of the data requested was in the past or in the future'


class OverlapsPastError(NotDuringError):
    """IndexError that indicates that there was an attempt to index some data that is in the PAST
    (i.e. is NO LONGER completely covered by the buffer)"""

    dlft_args = 'Some of the data requested is in the past'


class OverlapsFutureError(NotDuringError):
    """IndexError that indicates that there was an attempt to index some data that is in the FUTURE
    (i.e. is NOT YET completely covered by the buffer)"""

    dlft_args = 'Some of the data requested is in the future'


class RelationNotHandledError(TypeError):
    """TypeError that indicates that a relation is either not a valid one, or not handled by conditional clause."""


not_during_error = NotDuringError()
overlaps_past_error = OverlapsPastError()
overlaps_future_error = OverlapsFutureError()


# The IndexedBuffer, finally #############################################################################################


def _not_implemented(self, method_name, *args, **kwargs):
    raise NotImplementedError('')


# ram heavier, cpu lighter extend
def _extend_cpu_lighter_ram_heavier(self, iterable: Iterable) -> None:
    """Extend buffer with an iterable of items"""
    iterable = list(iterable)
    with self._lock:
        self._deque.extend(iterable)
        self.max_idx += len(iterable)


# cpu heavier, ram lighter extend
def _extend_ram_lighter_cpu_heavier(self, iterable: Iterable) -> None:
    """Extend buffer with an iterable of items"""
    c = count()
    counting_iter = map(itemgetter(0), zip(iterable, c))
    with self._lock:
        self._deque.extend(counting_iter)
        self.max_idx += next(c)


def none_safe_addition(x, y):
    """Adds the two numbers if x is not None, or return None if not"""
    if x is None:
        return None
    else:
        return x + y


def slice_args(slice_obj):
    return slice_obj.start, slice_obj.stop, slice_obj.step


def shift_slice(slice_obj, shift: Number):
    return slice(
        none_safe_addition(slice_obj.start, shift),
        none_safe_addition(slice_obj.stop, shift),
        slice_obj.step,
    )


def absolute_item(item, max_idx):
    """Returns an item with absolute references: i.e. with negative indices idx
    resolved to max_idx + idx

    >>> absolute_item(-1, 10)
    9
    >>> absolute_item(slice(-4, -2, 2), 10)
    slice(6, 8, 2)

    But anything else that's not a slice or int will be left untouched
    (and will probably result in errors if you use with IndexedBuffer)

    >>> absolute_item((-7, -2), 10)
    (-7, -2)

    """
    if isinstance(item, slice):
        start, stop, step = slice_args(item)
        if start is not None and start < 0:
            start = max_idx + start
        if stop is not None and stop < 0:
            stop = max_idx + stop
        return slice(start, stop, step)
    elif isinstance(item, int) and item < 0:
        return item + max_idx
    else:
        return item


class IndexedBuffer:
    """A list-like object that gives a limited-past read view of an unbounded stream

    For example, say we had the stream of increasing integers 0, 1, 2, ...
    that is being fed to indexedBuffer

    What IndexedBuffer(maxlen=4) offers is access to the buffer's contents,
    but using the indices that
    the stream (if it were one big list in memory) would use instead of the buffer's index.
        0 1 2 3 [4 5 6 7] 8 9

    IndexedBuffer uses collections.deque, exposing the append, extend,
    and clear methods, updating the index reference in a thread-safe manner.

    >>> s = IndexedBuffer(buffer_len=4)
    >>> s.extend(range(4))  # adding 4 elements in bulk (filling the buffer completely)
    >>> list(s)
    [0, 1, 2, 3]
    >>> s[2]
    2
    >>> s[1:2]
    [1]
    >>> s[1:1]
    []

    Let's add two more elements (using append this time), making the buffer "shift"

    >>> s.append(4)
    >>> s.append(5)
    >>> list(s)
    [2, 3, 4, 5]
    >>> s[2]
    2
    >>> s[5]
    5
    >>> s[2:5]
    [2, 3, 4]
    >>> s[3:6]
    [3, 4, 5]
    >>> assert s[2:6] == list(range(2, 6))

    You can slice with step:

    >>> s[2:6:2]
    [2, 4]

    You can slice with negatives
    >>> s[2:-2]
    [2, 3]

    On the other hand, if you ask for something that is not in the buffer (anymore, or yet), you'll get an
    error that tells you so:

    >>> # element for idx 1 is missing in [2, 3, 4, 5]
    >>> s[1:4]  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    OverlapsPastError: You asked for slice(1, 4, None), but the buffer only contains the index range: 2:6

    >>> # elements for 0:2 are missing (as well as 6:9, but OverlapsPastError trumps OverlapsFutureError
    >>> s[0:9]  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    OverlapsPastError: You asked for slice(0, 9, None), but the buffer only contains the index range: 2:6

    >>> # element for 6:9 are missing in [2, 3, 4, 5]
    >>> s[4:9]  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    OverlapsFutureError: You asked for slice(4, 9, None), but the buffer only contains the index range: 2:6
    """

    def __init__(
        self,
        buffer_len,
        prefill=(),
        if_overlaps_past=overlaps_past_error,
        if_overlaps_future=overlaps_future_error,
        slice_get_postproc: Callable = list,
    ):
        self._deque = deque(prefill, buffer_len)
        self.buffer_len = self._deque.maxlen
        self.max_idx = 0  # should correspond to the number of items added
        self.if_overlaps_past = if_overlaps_past
        self.if_overlaps_future = if_overlaps_future
        self.slice_get_postproc = slice_get_postproc
        self._lock = Lock()

    def __repr__(self):
        return (
            f'{type(self).__name__}('
            f'buffer_len={self.buffer_len}, '
            f'min_idx={self.min_idx}, '
            f'max_idx={self.max_idx}, ...)'
        )

    def __iter__(self):
        yield from self._deque

    @property
    def min_idx(self):
        return max(self.max_idx - self.buffer_len, 0)

    # TODO: Use singledispathmethod?
    def outer_to_buffer_idx(self, idx):
        if isinstance(idx, slice):
            return shift_slice(idx, -self.min_idx)
        elif isinstance(idx, int):
            if idx >= 0:
                return idx - self.min_idx
            else:  # idx < 0
                return self.buffer_len + idx
        elif isinstance(idx, Iterable):
            return tuple(x - self.min_idx for x in idx)
        else:
            raise TypeError(
                f'{type(idx)} are not handled. You requested the outer_to_buffer_idx of {idx}'
            )

    def __getitem__(self, item):
        item = absolute_item(
            item, self.max_idx
        )  # Note: Overhead for convenience of negative indices use (worth it?)
        relationship = simple_interval_relationship(
            item, (self.min_idx, self.max_idx + 1)
        )
        if relationship == Relations.DURING:
            item = self.outer_to_buffer_idx(item)
            if isinstance(item, slice):
                return self.slice_get_postproc(islice(self._deque, *slice_args(item)))
            else:
                try:
                    return self._deque[item]
                except IndexError:
                    if len(self._deque) < self.buffer_len:
                        raise self._overlaps_future_error(item)
                    else:
                        raise
        elif relationship == Relations.AFTER:
            raise self._overlaps_future_error(item)
        elif relationship == Relations.BEFORE:
            raise self._overlaps_past_error(item)
        else:
            raise RelationNotHandledError(
                f'The relation {relationship} is not handled.'
            )

    def _overlaps_past_error(self, item):
        return OverlapsPastError(
            f'You asked for {item}, but the buffer only contains the index range: {self.min_idx}:{self.max_idx}'
        )

    def _overlaps_future_error(self, item):
        return OverlapsFutureError(
            f'You asked for {item}, but the buffer only contains the index range: {self.min_idx}:{self.max_idx}'
        )

    def append(self, x) -> None:
        with self._lock:
            self._deque.append(x)
            self.max_idx += 1

    extend = _extend_ram_lighter_cpu_heavier

    def clear(self):
        with self._lock:
            self._deque.clear()
            self.max_idx = 0

    # # TODO: Sanity check.
    # def __len__(self):
    #     """Length in the sense of "number of items that passed through buffer so far -- not the length of the buff """
    #     return self.max_idx


def consume(gen, n):
    """Consume n iterations of generator (without returning elements)"""
    try:
        for _ in range(n):
            next(gen)
    except StopIteration:
        return None


from dataclasses import dataclass
from typing import Iterator


# TODO: Add some mechanism to deal with ITERABLE instead of just iterator. As it is we have some unwanted behavior with
#   iterables
@dataclass
class InfiniteSeq:
    """A list-like (read) view of an unbounded sequence/stream.

    It is the combination of `IndexedBuffer` and an iterator that will be used to
    source the buffer according to the slices that are requested.

    If a slice is requested whose data is "in the future", the iterator will be
    consumed until the buffer can satisfy that request.
    If the requested slice has any part of it that is "in the past", that is,
    has already been iterated through and is not in the buffer anymore, a
    `OverlapsPastError` will be raised.

    Therefore, `InfiniteSeq` is meant for ordered slice queries of size no more than
    the buffer size.
    If these conditions are satisfied, an `InfiniteSeq` will behave (with `i:j`
    queries) as if it were one long list in memory.

    Can be used with a live stream of data as long as the buffer size is big enough
    to handle the data production and query rates.

    For example, take an iterator that cycles from 0 to 99 forever:

    >>> from itertools import cycle
    >>> iterator = cycle(range(100))

    Let's make an `InfiniteSeq` instance for this stream, accomodating for a view of
    up to 11 items.

    >>> s = InfiniteSeq(iterator, buffer_len=11)

    Let's ask for element 15 (which is the (15 + 1)th element (and should have a value of 15).

    >>> s[15]
    15

    Now, to get this value, the iterator will move forward up to that point;
    that is, until the buffer's head (i.e. most recent) item contains that requested (15 + 1)th element.
    But the buffer is of size 11, so we still have access to a few previous elements:

    >>> s[11]
    11
    >>> s[5:15]
    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    But if we asked for anything before index 5...

    >>> s[2:7]  #doctest: +SKIP
    Traceback (most recent call last):
        ...
    OverlapsPastError: You asked for slice(2, 7, None), but the buffer only contains the index range: 5:16

    So we can't go backwards. But we can always go forwards:

    >>> s[95:105]
    [95, 96, 97, 98, 99, 0, 1, 2, 3, 4]

    You can also use slices with step and with negative integers (referencing the head of the buffer)

    >>> s[120:130:2]
    [20, 22, 24, 26, 28]
    >>> s[120:130]
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    >>> s[-8:-2]
    [22, 23, 24, 25, 26, 27]

    but you cannot slice farther back than the buffer

    >>> try:
    ...     s[-20:-2]
    ... except OverlapsPastError as e:
    ...     msg_text = str(e)
    >>> print(msg_text)
    You asked for slice(110, 128, None), but the buffer only contains the index range: 119:130

    Sometimes the source provides data in chunks. Sometimes these chunks are not even of fixed size.
    In those situations, you can use ``itertools.chain`` to "flatten" the iterator as in the following example:


    >>> from creek.infinite_sequence import InfiniteSeq
    >>> from collections import Mapping
    >>>
    >>> class Source(Mapping):
    ...     n = 100
    ...
    ...     __len__ = lambda self: self.n
    ...
    ...     def __iter__(self):
    ...         yield from range(self.n)
    ...
    ...     def __getitem__(self, k):
    ...         print(f"Asking for {k}")
    ...         return list(range(k * 10, (k + 1) * 10))
    ...
    >>>
    >>> source = Source()
    >>>

    See that when we ask for a chunk of data, there's a print notification about it.

    >>> assert source[3] == [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    Asking for 3

    Now let's make an iterator of the data and an InfiniteSeq (with buffer length 10) on top of it.

    >>> from itertools import chain
    >>> iterator = chain.from_iterable(source.values())
    >>> s = InfiniteSeq(iterator, 10)

    See that when you ask for :5, you see that chunk 0 is requested.

    >>> s[:5]
    Asking for 0
    [0, 1, 2, 3, 4]

    If you ask for something that's already in the buffer, you won't see the print notification though.

    >>> s[4:8]
    [4, 5, 6, 7]

    The following shows you how InfiniteSeq "hits" the data source as it's getting the data it needs for the request.

    >>> s[8:12]
    Asking for 1
    [8, 9, 10, 11]
    >>>
    >>> s[40:42]
    Asking for 2
    Asking for 3
    Asking for 4
    [40, 41]

    """

    iterator: Iterator
    buffer_len: int

    def __post_init__(self):
        self.indexed_buffer = IndexedBuffer(self.buffer_len)

    def __getitem__(self, item):
        if isinstance(item, slice):
            n_ticks_in_the_future = item.stop - self.indexed_buffer.max_idx
            if n_ticks_in_the_future > 0:
                # TODO: If indexed_buffer had a "fast-forward" (perhaps "peek")
                #  we could waste less buffer writes
                self.indexed_buffer.extend(islice(self.iterator, n_ticks_in_the_future))
                # consume(self.iterator, n_ticks_in_the_future)
            return self.indexed_buffer[item]
        elif isinstance(item, int):
            return self[slice(item, item + 1)][0]


def new_type(name, typ, doc=None):
    t = NewType(name, type)
    if doc is not None:
        t.__doc__ = doc
    return t


BufferInput = new_type(
    'BufferInput', Any, 'input_of/what_we_insert_in a buffer (before transformation)'
)

BufferItem = new_type(
    'BufferItem', Any, 'An item of a buffer (after input is transformed)'
)

InputDataTrans = new_type(
    'InputDataTrans',
    Callable[[BufferInput], BufferItem],
    'A function that transforms a BufferInput in to a BufferItem',
)
Query = new_type('Query', Any, 'A query (i.e. key, selection specification, etc.)')
QueryTrans = new_type(
    'QueryTrans',
    Callable[[Query], Query],
    'A function transforming a query into another, ready to be applied form',
)

FiltFunc = Callable[[BufferItem], bool]


def asis(obj):
    return obj


# TODO: Finish up and document
class BufferedGetter:
    """
    `BufferedGetter` is intended to be a more general (but not optimized) class that
    offers a query-interface to a buffer, intended to be used when the buffer is
    being filled by a (possibly live) stream of data items.

    By contrast...
    The `IndexedBuffer` is a particular case where the queries are slices and the index
    that is sliced on is an enumeration one.
    The `InfiniteSeq` is a class combining `IndexedBuffer` with a data source it can
    pull data from (according to the demands of the query).

    >>> from creek.infinite_sequence import BufferedGetter
    >>>
    >>>
    >>> b = BufferedGetter(20)
    >>> b.extend([
    ...     (1, 3, 'completely before'),
    ...     (2, 4, 'still completely before (upper bounds are strict)'),
    ...     (3, 6, 'partially before, but overlaps bottom'),
    ...     (4, 5, 'totally', 'inside'),  # <- note this tuple has 4 elements
    ...     (5, 8),  # <- note this tuple has only the minimum (2) elements,
    ...     (7, 10, 'partially after, but overlaps top'),
    ...     (8, 11, 'completely after (strict upper bound)'),
    ...     (100, 101, 'completely after (obviously)')
    ... ])
    >>> b[lambda x: 3 < x[0] < 8]  # doctest: +NORMALIZE_WHITESPACE
    [(4, 5, 'totally', 'inside'),
     (5, 8),
     (7, 10, 'partially after, but overlaps top')]
    """

    def __init__(
        self,
        buffer_len,
        prefill=(),
        input_data_trans=asis,
        query_trans=asis,
        # if_overlaps_past=overlaps_past_error,
        # if_overlaps_future=overlaps_future_error,
        slice_get_postproc: Callable = list,
    ):
        self._deque = deque(prefill, buffer_len)
        self.buffer_len = self._deque.maxlen
        self.input_data_trans = input_data_trans
        self.query_trans = query_trans
        # self.max_idx = 0  # should correspond to the number of items added
        # self.if_overlaps_past = if_overlaps_past
        # self.if_overlaps_future = if_overlaps_future
        self.slice_get_postproc = slice_get_postproc
        self._lock = Lock()

    def ingress(self, x: BufferInput) -> BufferItem:
        return x

    def append(self, x: BufferInput) -> None:
        with self._lock:
            x = self.input_data_trans(x)
            self._deque.append(x)

    def extend(self, iterable):
        for x in iterable:
            self.append(x)

    def clear(self):
        with self._lock:
            self._deque.clear()

    def __getitem__(self, q: Query):
        return self._getitem(q)

    def _getitem(self, q: Query):
        # note: just a composition
        return self.slice_get_postproc(self.filter(self.query_trans(q)))

    def filter(self, filt: Callable):
        # assert callable(filt), f'filt must be callable, was: {filt}'
        return filter(filt, self._deque)

    def __iter__(self):  # to dodge the iteration falling back to __getitem__(i)
        yield from self._deque
