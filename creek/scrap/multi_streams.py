from creek.multi_streams import *

from warnings import warn

warn(f'Moved to creek.multi_streams')

from dataclasses import dataclass

from typing import Any, NewType, Callable, Iterable, Union, Tuple
from numbers import Number


def MyType(
    name: str,
    tp: type = Any,
    doc: Optional[str] = None,
    aka: Optional[Union[str, Iterable[str]]] = None,
):
    """Like `typing.NewType` with some extras (`__doc__` and `_aka` attributes, etc.)
    """

    new_tp = NewType(name, tp)
    if doc is not None:
        new_tp.__doc__ = doc
    if aka is not None:
        new_tp._aka = aka
    return new_tp


TimeIndex = MyType(
    'TimeIndex',
    Number,
    doc='A number indexing time. Could be in an actual time unit, or could just be '
    'an enumerator (i.e. "ordinal time")',
)
BT = MyType(
    'BT',
    TimeIndex,
    doc='TimeIndex for the lower bound of an interval of time. '
    'Stands for "Bottom Time". By convention, a BT is inclusive.',
)
TT = MyType(
    'TT',
    TimeIndex,
    doc='TimeIndex for the upper bound of an interval of time. '
    'Stands for "Upper Time". By convention, a TT is exclusive.',
)
IntervalTuple = MyType(
    'IntervalTuple',
    Tuple[BT, TT],
    doc='Denotes an interval of time by specifying the (BT, TT) pair',
)
IntervalSlice = MyType(
    'IntervalSlice',
    slice,  # Note: extra condition: non-None .start and .end, and no .step
    doc='Denotes an interval of time by specifying the (BT, TT) pair',
)

Intervals = Iterable[IntervalTuple]
BufferedIntervals = Intervals
RetrievedIntervals = Intervals
QueryInterval = Intervals
IntervalsRetriever = Callable[[QueryInterval, BufferedIntervals], RetrievedIntervals]


@dataclass
class IntervalSlicer:
    match_intervals: IntervalsRetriever
