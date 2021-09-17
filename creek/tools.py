"""Tools to work with creek objects"""

from typing import Union, Any, Callable
from dataclasses import dataclass


Number = Union[int, float]
Index = Number
DataItem = Any
IndexUpdater = Callable[[Index, DataItem], Index]


def count_increments(current_idx, obj, step=1):
    return current_idx + step


def size_increments(current_idx, obj, size_func=len):
    return current_idx + size_func(obj)


@dataclass
class DynamicIndexer:
    """
    :param current_idx: The index to start at (the first data item will have this index)
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

    To get the index starting at 10, we can specify ``start_idx=10``, and to step the
    index by 3 we can partialize ``count_increments``:

    >>> from functools import partial
    >>> step3 = partial(count_increments, step=3)
    >>> list(map(DynamicIndexer(start_idx=10, idx_updater=step3), stream))
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

    start_idx: Index = 0
    idx_updater: IndexUpdater = count_increments

    count_increments = staticmethod(count_increments)
    size_increments = staticmethod(size_increments)

    def __post_init__(self):
        self.current_idx = self.start_idx

    def __call__(self, x):
        _current_idx = self.current_idx
        self.current_idx = self.idx_updater(_current_idx, x)
        return _current_idx, x
