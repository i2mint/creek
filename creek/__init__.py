"""creek: a simple facade for streams.

Wrap a stream in a ``Creek`` and override ``pre_iter``, ``data_to_obj`` or
``post_iter`` to layer transformations on it; read an unbounded stream as if it
were a list with ``InfiniteSeq``; index, window and segment streams with the
functions and classes of ``creek.tools``.

Main entry points:

- ``Creek``: the layer-able stream wrapper (``creek.base``)
- ``InfiniteSeq, IndexedBuffer``: list-like views of unbounded streams (``creek.infinite_sequence``)
- ``BufferStats, Segmenter``: rolling-window statistics and segmentation (``creek.tools``)
- ``dynamically_index, filter_and_index_stream``: indexing streams (``creek.tools``)

>>> from creek import Creek
>>> class Doubler(Creek):
...     def data_to_obj(self, x):
...         return x * 2
>>> list(Doubler([1, 2, 3]))
[2, 4, 6]
"""

from creek.base import Creek

from creek.infinite_sequence import InfiniteSeq, IndexedBuffer
from creek.tools import (
    filter_and_index_stream,
    dynamically_index,
    DynamicIndexer,
    count_increments,
    size_increments,
    BufferStats,
    Segmenter,
)
