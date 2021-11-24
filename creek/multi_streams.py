"""Tools for multi-streams"""

from itertools import product
from typing import Mapping, Iterable, Any, Optional, Callable
import heapq
from dataclasses import dataclass
from operator import itemgetter

from creek.util import Pipe, identity_func

StreamsMap = Mapping[Any, Iterable]  # a map of {name: stream} pairs


@dataclass
class MergedStreams:
    """Creates an iterable of (stream_id, stream_item) pairs from a stream Mapping,
    that is, {stream_id: stream, ...}.

    The stream_item will be yield in sorted order.
    Sort behavior can be modified by the ``sort_key`` argument which behaves like ``key``
    arguments of built-in like ``sorted``, ``heapq.merge``, ``itertools.groupby``, etc.

    If given, the `sort_key` function applies to ``stream_item`` (not to ``stream_id``).

    Important: To function as expected, the streams should be already sorted (according
    to the ``sort_key`` order).

    The cannonical use case of this function is to "flatten", or "weave together"
    multiple streams of timestamped data. We're given several streams that provide
    ``(timestamp, data)`` items (where timestamps arrive in order within each stream)
    and we get a single stream of ``(stream_id, (timestamp, data))`` items where
    the ``timestamp``s are yield in sorted order.

    The following example uses a dict pointing to a fixed-size list as the ``stream_map``
    but in general the ``stream_map`` will be a ``Mapping`` (not necessarily a dict)
    whose values are potentially bound-less streams.

    >>> streams_map = {
    ...     'hello': [(2, 'two'), (3, 'three'), (5, 'five')],
    ...     'world': [(0, 'zero'), (1, 'one'), (3, 'three'), (6, 'six')]
    ... }
    >>> streams_items = MergedStreams(streams_map)
    >>> it = iter(streams_items)
    >>> list(it)  # doctest: +NORMALIZE_WHITESPACE
    [('world', (0, 'zero')),
     ('world', (1, 'one')),
     ('hello', (2, 'two')),
     ('hello', (3, 'three')),
     ('world', (3, 'three')),
     ('hello', (5, 'five')),
     ('world', (6, 'six'))]
    """

    streams_map: StreamsMap
    sort_key: Optional[Callable] = None

    def __post_init__(self):
        if self.sort_key is None:
            self.effective_sort_key = itemgetter(1)
        else:
            self.effective_sort_key = Pipe(itemgetter(1), self.sort_key)

    def __iter__(self):
        for item in heapq.merge(
            *multi_stream_items(self.streams_map), key=self.effective_sort_key
        ):
            yield item


def multi_stream_items(streams_map: StreamsMap):
    """Provides a iterable of (k1, v1_1), (k1, v1_2), ...
    >>> streams_map = {'hello': 'abc', 'world': [1, 2]}
    >>> hello_items, world_items = multi_stream_items(streams_map)
    >>> list(hello_items)
    [('hello', 'a'), ('hello', 'b'), ('hello', 'c')]
    >>> list(world_items)
    [('world', 1), ('world', 2)]
    """
    for stream_id, stream in streams_map.items():
        yield product([stream_id], stream)


class SortKeys:
    all_but_last = staticmethod(itemgetter(-1))
