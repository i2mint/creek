"""Tools to work with creek objects"""

import time
from typing import Union, Any, Callable, Tuple, TypeVar, Iterable
from dataclasses import dataclass

Index = Any
DataItem = TypeVar('DataItem')
# TODO: Could have more args. How to specify this in typing?
IndexUpdater = Callable[[Index, DataItem], Index]
Indexer = Callable[[DataItem], Tuple[Index, DataItem]]


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

from functools import partial
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
