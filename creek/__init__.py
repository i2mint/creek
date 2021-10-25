"""main creek objects"""
from creek.base import Creek

from creek.infinite_sequence import InfiniteSeq, IndexedBuffer
from creek.tools import (
    filter_and_index_stream,
    dynamically_index,
    DynamicIndexer,
    count_increments,
    size_increments,
)
