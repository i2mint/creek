"""Tools to make annotators.

"""

from typing import Union, Iterable, Tuple, Callable, KT, VT
from operator import itemgetter
from functools import partial
from i2 import Pipe

# ------------------- Types -------------------
# TODO: Should we use the term KV (key-value-pair) instead of annotation?
# TODO: Should we use some Time (numerical) type instead of KT here?
IndexAnnot = Tuple[KT, VT]
IndexAnnot.__doc__ = """An annotation whose key is an (time) index. 
KT is usually numerical and represents time. 
VT holds the value (info) of the annotation.
"""

Interval = Tuple[KT, KT]  # note this it's two KTs here, usually numerical.
Interval.__doc__ = """An interval; i.e. a pair of indices"""

IntervalAnnot = Tuple[Interval, VT]
IntervalAnnot.__doc__ = """An annotation whose key is an interval."""

KvExtractor = Callable[[Iterable], Iterable[IndexAnnot]]
IntervalAnnot.__doc__ = """A function that extracts annotations from an iterable."""

FilterFunc = Callable[..., bool]


def always_true(x):
    return True


# ------------------- Annotators -------------------
def track_intervals(
    indexed_tags: Iterable[IndexAnnot], track_tag: FilterFunc = always_true
) -> Iterable[IntervalAnnot]:
    """Track intervals of tags in an iterable of indexed tags.

    Example usage:

    >>> iterable = ['a', 'b', 'a', 'b', 'c', 'c', 'd', 'd']
    >>> list(track_intervals(enumerate(iterable)))
    [((0, 2), 'a'), ((1, 3), 'b'), ((4, 5), 'c'), ((6, 7), 'd')]
    >>> list(track_intervals(enumerate(iterable), track_tag=lambda x: x in {'a', 'd'}))
    [((0, 2), 'a'), ((6, 7), 'd')]

    """
    open_tags = {}
    for index, tag in indexed_tags:
        if track_tag(tag):
            if tag not in open_tags:
                open_tags[tag] = index
            else:
                yield ((open_tags[tag], index), tag)
                del open_tags[tag]


def mk_interval_extractor(
    *,
    kv_extractor: KvExtractor = enumerate,
    include_tag: bool = True,
    track_tag: FilterFunc = always_true
) -> Callable[[Iterable], Iterable[Union[Interval, IntervalAnnot]]]:
    """Make an interval extractor from a key-value extractor.

    Example usage:

    >>> iterable = ['a', 'b', 'a', 'b', 'c', 'c', 'd', 'd']
    >>> extract_intervals = mk_interval_extractor()
    >>> list(extract_intervals(iterable))
    [((0, 2), 'a'), ((1, 3), 'b'), ((4, 5), 'c'), ((6, 7), 'd')]
    >>> extract_intervals_without_tags = mk_interval_extractor(include_tag=False)
    >>> list(extract_intervals_without_tags(iterable))
    [(0, 2), (1, 3), (4, 5), (6, 7)]

    See:

    """
    interval_extractor = Pipe(
        kv_extractor, partial(track_intervals, track_tag=track_tag)
    )
    if not include_tag:
        only_interval = partial(map, itemgetter(0))
        interval_extractor = Pipe(interval_extractor, only_interval)
    return interval_extractor
