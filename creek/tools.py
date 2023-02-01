"""Tools to work with creek objects"""

import time
from collections import deque
from typing import (
    Any,
    Callable,
    Tuple,
    TypeVar,
    Callable,
    Any,
    Iterable,
    Sequence,
    cast,
)
from dataclasses import dataclass
from itertools import chain
from operator import itemgetter
from functools import partial

from creek.util import Pipe

Index = Any
DataItem = TypeVar('DataItem')
# TODO: Could have more args. How to specify this in typing?
IndexUpdater = Callable[[Index, DataItem], Index]
Indexer = Callable[[DataItem], Tuple[Index, DataItem]]

from itertools import chain


# TODO: Possible performance enhancement by using class with precompiled slices
# TODO: Compare with apply_and_fanout (no cast here, and slice instead of :)
# Note: I hesitated to make the signature (func, apply_to_idx, seq) instead
# Note: This would have allowed to use partial(apply_func_to_index, func, idx)
# Note: instead, but NOT partial(apply_func_to_index, func=func, apply_to_idx=idx)!!
def apply_func_to_index(seq, apply_to_idx, func):
    """
    >>> apply_func_to_index([1, 2, 3], 1, lambda x: x * 10)
    (1, 20, 3)

    If you're going to apply the same function to the same index, you might
    want to partialize ``apply_func_to_index`` to be able to reuse it simply:

    >>> from functools import partial
    >>> f = partial(apply_func_to_index, apply_to_idx=0, func=str.upper)
    >>> list(map(f, ['abc', 'defgh']))
    [('A', 'b', 'c'), ('D', 'e', 'f', 'g', 'h')]

    """
    apply_to_element, *_ = seq[slice(apply_to_idx, apply_to_idx + 1)]
    return tuple(
        chain(
            seq[slice(None, apply_to_idx)],
            [func(apply_to_element)],
            seq[slice(apply_to_idx + 1, None)],
        )
    )


def apply_and_fanout(
    seq: Sequence, func: Callable[[Any], Iterable], idx: int
) -> Iterable[tuple]:
    """Apply function (that returns an Iterable) to an element of a sequence
    and fanout (broadcast) the resulting items, to produce an iterable of tuples each
    containing
    one of these items along with 'a copy' of the other tuple elements

    >>> list(apply_and_fanout([1, 'abc', 3], iter, 1))
    [(1, 'a', 3), (1, 'b', 3), (1, 'c', 3)]
    >>> list(apply_and_fanout(['bob', 'alice', 2], lambda x: x * ['hi'], 2))
    [('bob', 'alice', 'hi'), ('bob', 'alice', 'hi')]
    >>> list(apply_and_fanout(["bob", "alice", 2], lambda x: x.upper(), 1))
    [('bob', 'A', 2), ('bob', 'L', 2), ('bob', 'I', 2), ('bob', 'C', 2), ('bob', 'E', 2)]

    See Also:
        ``fanout_and_flatten`` and ``fanout_and_flatten_dicts``
    """
    seq = tuple(seq)  # TODO: Overhead: Should we impose seq to be tuple?
    # TODO: See how apply_func_to_index takes care of this problem with chain
    left_seq = seq[0 : max(idx, 0)]  # TODO: Use seq[None:idx] instead?
    right_seq = seq[(idx + 1) :]
    return (left_seq + (item,) + right_seq for item in func(seq[idx]))


def fanout_and_flatten(iterable_of_seqs, func, idx, aggregator=chain.from_iterable):
    """Apply apply_and_fanout to an iterable of sequences.

    >>> seq_iterable = [('abcdef', 'first'), ('ghij', 'second')]
    >>> func = lambda a: zip(*([iter(a)] * 2))  # func is a chunker
    >>> assert list(fanout_and_flatten(seq_iterable, func, 0)) == [
    ...     (('a', 'b'), 'first'),
    ...     (('c', 'd'), 'first'),
    ...     (('e', 'f'), 'first'),
    ...     (('g', 'h'), 'second'),
    ...     (('i', 'j'), 'second')
    ... ]
    """
    apply = partial(apply_and_fanout, func=func, idx=idx)
    return aggregator(map(apply, iterable_of_seqs))


def fanout_and_flatten_dicts(
    iterable_of_dicts, func, fields, idx_field, aggregator=chain.from_iterable
):
    """Apply apply_and_fanout to an iterable of dicts.

    >>> iterable_of_dicts = [
    ...     {'wf': 'abcdef', 'tag': 'first'}, {'wf': 'ghij', 'tag': 'second'}
    ... ]
    >>> func = lambda a: zip(*([iter(a)] * 2))  # func is a chunker
    >>> fields = ['wf', 'tag']
    >>> idx_field = 'wf'
    >>> assert list(
    ...     fanout_and_flatten_dicts(iterable_of_dicts, func, fields, idx_field)) == [
    ...         {'wf': ('a', 'b'), 'tag': 'first'},
    ...         {'wf': ('c', 'd'), 'tag': 'first'},
    ...         {'wf': ('e', 'f'), 'tag': 'first'},
    ...         {'wf': ('g', 'h'), 'tag': 'second'},
    ...         {'wf': ('i', 'j'), 'tag': 'second'}
    ... ]
    """
    egress = Pipe(partial(zip, fields), dict)
    return map(
        egress,
        fanout_and_flatten(
            map(itemgetter(*fields), iterable_of_dicts),
            func=func,
            idx=fields.index(idx_field),
            aggregator=aggregator,
        ),
    )


def filter_and_index_stream(
    stream: Iterable, data_item_filt, timestamper: Indexer = enumerate
):
    """Index a stream and filter it (based only on the data items).

    >>> assert (
    ... list(filter_and_index_stream('this  is   a   stream', data_item_filt=' ')) == [
    ... (0, 't'),
    ... (1, 'h'),
    ... (2, 'i'),
    ... (3, 's'),
    ... (6, 'i'),
    ... (7, 's'),
    ... (11, 'a'),
    ... (15, 's'),
    ... (16, 't'),
    ... (17, 'r'),
    ... (18, 'e'),
    ... (19, 'a'),
    ... (20, 'm')
    ... ])

    >>> list(filter_and_index_stream(
    ...     [1, 2, 3, 4, 5, 6, 7, 8],
    ...     data_item_filt=lambda x: x % 2))
    [(0, 1), (2, 3), (4, 5), (6, 7)]
    """
    if not callable(data_item_filt):
        sentinel = data_item_filt
        data_item_filt = lambda x: x != sentinel
    return filter(lambda x: data_item_filt(x[1]), timestamper(stream))


# TODO: Refactor dynamic indexing set up so that
#  IndexUpdater = Callable[[DataItem, Index, ...], Index] (data and index inversed)
#  Rationale: One can (and usually would want to) have a default current_idx, which
#  can be used as the start index too.
count_increments: IndexUpdater


def count_increments(current_idx: Index, obj: DataItem, step=1):
    return current_idx + step


size_increments: IndexUpdater


def size_increments(current_idx, obj: DataItem, size_func=len):
    return current_idx + size_func(obj)


current_time: IndexUpdater


def current_time(current_idx, obj):
    """Doesn't even look at current_idx or obj. Just gives the current time"""
    return time.time()


@dataclass
class DynamicIndexer:
    """
    :param start: The index to start at (the first data item will have this index)
    :param idx_updater: The (Index, DataItem) -> Index

    Let's take a finite stream of finite iterables (strings here):

    >>> stream = ['stream', 'of', 'different', 'sized', 'chunks']

    The default ``DynamicIndexer`` just does what ``enumerate`` does:

    >>> counter_index = DynamicIndexer()
    >>> list(map(counter_index, stream))
    [(0, 'stream'), (1, 'of'), (2, 'different'), (3, 'sized'), (4, 'chunks')]

    That's because it uses the default ``idx_updater`` function just increments by one.
    This function, DynamicIndexer.count_increments, is shown below

    >>> def count_increments(current_idx, data_item, step=1):
    ...     return current_idx + step

    To get the index starting at 10, we can specify ``start=10``, and to step the
    index by 3 we can partialize ``count_increments``:

    >>> from functools import partial
    >>> step3 = partial(count_increments, step=3)
    >>> list(map(DynamicIndexer(start=10, idx_updater=step3), stream))
    [(10, 'stream'), (13, 'of'), (16, 'different'), (19, 'sized'), (22, 'chunks')]

    You can specify any custom ``idx_updater`` you want: The requirements being that
    this function should take ``(current_idx, data_item)`` as the input, and
    return the next "current index", that is, what the index of the next data item will
    be.
    Note that ``count_increments`` ignored the ``data_item`` completely, but sometimes
    you want to take the data item into account.
    For example, your data item may contain several elements, and you want your
    index to index these elements, therefore you should update your index by
    incrementing it with the number of elements.

    We have ``DynamicIndexer.size_increments`` for that, the code is shown below:

    >>> def size_increments(current_idx, data_item, size_func=len):
    ...     return current_idx + size_func(data_item)
    >>> size_index = DynamicIndexer(idx_updater=DynamicIndexer.size_increments)
    >>> list(map(size_index, stream))
    [(0, 'stream'), (6, 'of'), (8, 'different'), (17, 'sized'), (22, 'chunks')]

    Q: What if I want the index of a data item to be a function of the data item itself?

    A: Then you would use that function to make the ``(idxof(data_item), data_item)``
    pairs directly. ``DynamicIndexer`` is for the use case where the index of an item
    depends on the (number of, sizes of, etc.) items that came before it.

    """

    start: Index = 0
    idx_updater: IndexUpdater = count_increments

    count_increments = staticmethod(count_increments)
    size_increments = staticmethod(size_increments)

    def __post_init__(self):
        self.current_idx = self.start

    def __call__(self, x):
        _current_idx = self.current_idx
        self.current_idx = self.idx_updater(_current_idx, x)
        return _current_idx, x


def dynamically_index(iterable: Iterable, start=0, idx_updater=count_increments):
    """Generalization of `enumerate(iterable)` that allows one to specify how the
    indices should be updated.

    The default is the sae behavior as `enumerate`: Starts with 0 and increments by 1.

    >>> stream = ['stream', 'of', 'different', 'sized', 'chunks']
    >>> assert (list(dynamically_index(stream, start=2))
    ...     == list(enumerate(stream, start=2))
    ...     == [(2, 'stream'), (3, 'of'), (4, 'different'), (5, 'sized'), (6, 'chunks')]
    ... )

    Say we wanted to increment the indices according to the size of the last item
    instead of just incrementing by 1 at every iteration tick...

    >>> def size_increments(current_idx, data_item, size_func=len):
    ...     return current_idx + size_func(data_item)
    >>> size_index = DynamicIndexer(idx_updater=DynamicIndexer.size_increments)
    >>> list(map(size_index, stream))
    [(0, 'stream'), (6, 'of'), (8, 'different'), (17, 'sized'), (22, 'chunks')]

    """
    dynamic_indexer = DynamicIndexer(start, idx_updater)
    return map(dynamic_indexer, iterable)


# Alternative to the above implementation:

from itertools import accumulate


def _dynamic_indexer(stream, idx_updater: IndexUpdater = count_increments, start=0):
    index_func = partial(accumulate, func=idx_updater, initial=start)
    obj = zip(index_func(stream), stream)
    return obj


def alt_dynamically_index(idx_updater: IndexUpdater = count_increments, start=0):
    """Alternative to dynamically_index using itertools and partial

    >>> def size_increments(current_idx, data_item, size_func=len):
    ...     return current_idx + size_func(data_item)
    ...
    >>> stream = ['stream', 'of', 'different', 'sized', 'chunks']
    >>> indexer = alt_dynamically_index(size_increments)
    >>> t = list(indexer(stream))
    >>> assert t == [(0, 'stream'), (6, 'of'), (8, 'different'), (17, 'sized'),
    ...              (22, 'chunks')]
    """
    return partial(_dynamic_indexer, idx_updater=idx_updater, start=start)


# ---------------------------------------------------------------------------------------
# Slicing index segment streams


def segment_overlaps(bt_tt_segment, query_bt, query_tt):
    """Returns True if, and only if, bt_tt_segment overlaps query interval.

    A `bt_tt_segment` will need to be of the ``(bt, tt, *data)`` format.
    That is, an iterable of at least two elements (the ``bt`` and ``tt``) followed with
    more elements (the actual segment data).

    This function is made to be curried, as shown in the following example:

    >>> from functools import partial
    >>> overlapping_segments_filt = partial(segment_overlaps, query_bt=4, query_tt=8)
    >>>
    >>> list(filter(overlapping_segments_filt, [
    ...     (1, 3, 'completely before'),
    ...     (2, 4, 'still completely before (upper bounds are strict)'),
    ...     (3, 6, 'partially before, but overlaps bottom'),
    ...     (4, 5, 'totally', 'inside'),  # <- note this tuple has 4 elements
    ...     (5, 8),  # <- note this tuple has only the minimum (2) elements,
    ...     (7, 10, 'partially after, but overlaps top'),
    ...     (8, 11, 'completely after (strict upper bound)'),
    ...     (100, 101, 'completely after (obviously)')
    ... ]))  # doctest: +NORMALIZE_WHITESPACE
    [(3, 6, 'partially before, but overlaps bottom'),
    (4, 5, 'totally', 'inside'),
    (5, 8),
    (7, 10, 'partially after, but overlaps top')]

    """
    bt, tt, *segment = bt_tt_segment
    return (
        query_bt < tt <= query_tt  # the top part of segment intersects
        or query_bt <= bt < query_tt  # the bottom part of the segment intersects
        # If it's both, the interval is entirely inside the query
    )


Stats = Any
_no_value_specified_sentinel = cast(int, object())


class BufferStats(deque):
    """A callable (fifo) buffer. Calls add input to it, but also returns some results
    computed from it's contents.

    What "add" means is configurable (through ``add_new_val`` arg). Default
    is append, but can be extend etc.

    >>> bs = BufferStats(maxlen=4, func=sum)
    >>> list(map(bs, range(7)))
    [0, 1, 3, 6, 10, 14, 18]

    See what happens when you feed the same sequence again:

    >>> list(map(bs, range(7)))
    [15, 12, 9, 6, 10, 14, 18]

    More examples:

    >>> list(map(BufferStats(maxlen=4, func=''.join), 'abcdefgh'))
    ['a', 'ab', 'abc', 'abcd', 'bcde', 'cdef', 'defg', 'efgh']

    >>> from math import prod
    >>> list(map(BufferStats(maxlen=4, func=prod), range(7)))
    [0, 0, 0, 0, 24, 120, 360]

    With a different ``add_new_val`` choice.

    >>> bs = BufferStats(maxlen=4, func=''.join, add_new_val=deque.appendleft)
    >>> list(map(bs, 'abcdefgh'))
    ['a', 'ba', 'cba', 'dcba', 'edcb', 'fedc', 'gfed', 'hgfe']

    With ``add_new_val=deque.extend``, data can be fed in chunks.
    In the following, also see how we use iterize to get a function that
    takes an iterator and returns an iterator

    >>> from creek.util import iterize
    >>> window_stats = iterize(BufferStats(
    ...     maxlen=4, func=''.join, add_new_val=deque.extend)
    ... )
    >>> chks = ['a', 'bc', 'def', 'gh']
    >>> for x in window_stats(chks):
    ...     print(x)
    a
    abc
    cdef
    efgh

    Note: To those who might think that they can optimize this for special
    cases: Yes you can.
    But SHOULD you? Is it worth the increase in complexity and reduction in
    flexibility?
    See https://github.com/thorwhalen/umpyre/blob/master/misc/performance_of_rolling_window_stats.md

    """

    # __name__ = 'BufferStats'

    def __init__(
        self,
        values=(),
        maxlen: int = _no_value_specified_sentinel,
        func: Callable = sum,
        add_new_val: Callable = deque.append,
    ):
        """

        :param maxlen: Size of the buffer
        :param func: The function to be computed (on buffer contents) and
        returned when buffer is "called"
        :param add_new_val: The function that adds values on the buffer.
        Signature must be (self, new_val)
            Is usually a deque method (``deque.append`` by default, but could
            be ``deque.extend``, ``deque.appendleft`` etc.).
            Can also be any other function that
            has a valid (self, new_val) signature.
        """
        if maxlen is _no_value_specified_sentinel:
            raise TypeError('You are required to specify maxlen')
        if not isinstance(maxlen, int):
            raise TypeError(f'maxlen must be an integer, was: {maxlen}')

        super().__init__(values, maxlen=maxlen)
        self.func = func
        if isinstance(add_new_val, str):
            # assume add_new_val is a method of deque:
            add_new_val = getattr(self, add_new_val)
        self.add_new_val = add_new_val
        self.__name__ = 'BufferStats'

    def __call__(self, new_val) -> Stats:
        self.add_new_val(self, new_val)  # add the new value
        return self.func(self)


def is_not_none(x):
    return x is not None


def return_buffer_on_stats_condition(
    stats: Stats, buffer: Iterable, cond: Callable = is_not_none, else_val=None
):
    """

    >>> return_buffer_on_stats_condition(
    ... stats=3, buffer=[1,2,3,4], cond=lambda x: x%2 == 1
    ... )
    [1, 2, 3, 4]
    >>> return_buffer_on_stats_condition(
    ... stats=3, buffer=[1,2,3,4], cond=lambda x: x%2 == 0, else_val='3 is not even!'
    ... )
    '3 is not even!'
    """

    if cond(stats):
        return buffer
    else:
        return else_val


@dataclass
class Segmenter:
    """

    >>> gen = iter(range(200))
    >>> bs = BufferStats(maxlen=10, func=sum)
    >>> return_if_stats_is_odd = partial(
    ...     return_buffer_on_stats_condition,
    ...     cond=lambda x: x%2 == 1, else_val='The sum is not odd!'
    ... )
    >>> seg = Segmenter(buffer=bs, stats_buffer_callback=return_if_stats_is_odd)

    Since the sum of the values in the buffer [1] is odd, the buffer is returned:

    >>> seg(new_val=1)
    [1]

    Adding ``1 + 2`` is still odd so:

    >>> seg(new_val=2)
    [1, 2]

    Now since ``1 + 2 + 5`` is even, the ``else_val`` of ``return_if_stats_is_odd``
    is returned instead

    >>> seg(new_val=5)
    'The sum is not odd!'
    """

    buffer: BufferStats
    stats_buffer_callback: Callable[
        [Stats, Iterable], Any
    ] = return_buffer_on_stats_condition
    __name__ = 'Segmenter'

    def __call__(self, new_val):
        stats = self.buffer(new_val)
        return self.stats_buffer_callback(stats, list(self.buffer))
