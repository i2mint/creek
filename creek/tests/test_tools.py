"""Unit tests for creek.tools module."""
import pytest
from functools import partial
from math import prod
from creek.tools import (
    filter_and_index_stream,
    dynamically_index,
    DynamicIndexer,
    count_increments,
    size_increments,
    BufferStats,
    Segmenter,
    segment_overlaps,
    return_buffer_on_stats_condition,
)


class TestFilterAndIndexStream:
    """Tests for filter_and_index_stream function."""

    def test_filter_with_sentinel_value(self):
        """Test filtering with a sentinel value."""
        result = list(
            filter_and_index_stream('this  is   a   stream', data_item_filt=' ')
        )
        expected = [
            (0, 't'), (1, 'h'), (2, 'i'), (3, 's'),
            (6, 'i'), (7, 's'),
            (11, 'a'),
            (15, 's'), (16, 't'), (17, 'r'), (18, 'e'), (19, 'a'), (20, 'm')
        ]
        assert result == expected

    def test_filter_with_callable(self):
        """Test filtering with a callable predicate."""
        result = list(
            filter_and_index_stream(
                [1, 2, 3, 4, 5, 6, 7, 8],
                data_item_filt=lambda x: x % 2
            )
        )
        expected = [(0, 1), (2, 3), (4, 5), (6, 7)]
        assert result == expected

    def test_empty_stream(self):
        """Test with empty stream."""
        result = list(filter_and_index_stream([], data_item_filt=lambda x: True))
        assert result == []

    def test_no_matches(self):
        """Test when no items match filter."""
        result = list(
            filter_and_index_stream([1, 2, 3], data_item_filt=lambda x: x > 10)
        )
        assert result == []


class TestCountIncrements:
    """Tests for count_increments function."""

    def test_default_step(self):
        """Test count_increments with default step of 1."""
        assert count_increments(0, 'item') == 1
        assert count_increments(5, 'item') == 6
        assert count_increments(99, 'item') == 100

    def test_custom_step(self):
        """Test count_increments with custom step."""
        assert count_increments(0, 'item', step=5) == 5
        assert count_increments(10, 'item', step=3) == 13
        assert count_increments(0, 'item', step=-1) == -1


class TestSizeIncrements:
    """Tests for size_increments function."""

    def test_default_size_func(self):
        """Test size_increments with default len function."""
        assert size_increments(0, 'hello') == 5
        assert size_increments(5, [1, 2, 3]) == 8
        assert size_increments(0, '') == 0

    def test_custom_size_func(self):
        """Test size_increments with custom size function."""
        custom_size = lambda x: x  # size is the value itself
        assert size_increments(0, 10, size_func=custom_size) == 10
        assert size_increments(5, 3, size_func=custom_size) == 8


class TestDynamicIndexer:
    """Tests for DynamicIndexer class."""

    def test_default_behavior_like_enumerate(self):
        """Test that default DynamicIndexer behaves like enumerate."""
        stream = ['stream', 'of', 'different', 'sized', 'chunks']
        indexer = DynamicIndexer()
        result = list(map(indexer, stream))
        expected = list(enumerate(stream))
        assert result == expected

    def test_custom_start(self):
        """Test DynamicIndexer with custom start value."""
        stream = ['a', 'b', 'c']
        indexer = DynamicIndexer(start=10)
        result = list(map(indexer, stream))
        assert result == [(10, 'a'), (11, 'b'), (12, 'c')]

    def test_with_step(self):
        """Test DynamicIndexer with custom step."""
        stream = ['stream', 'of', 'different', 'sized', 'chunks']
        step3 = partial(count_increments, step=3)
        indexer = DynamicIndexer(start=10, idx_updater=step3)
        result = list(map(indexer, stream))
        assert result == [
            (10, 'stream'), (13, 'of'), (16, 'different'),
            (19, 'sized'), (22, 'chunks')
        ]

    def test_with_size_increments(self):
        """Test DynamicIndexer with size-based increments."""
        stream = ['stream', 'of', 'different', 'sized', 'chunks']
        indexer = DynamicIndexer(idx_updater=DynamicIndexer.size_increments)
        result = list(map(indexer, stream))
        assert result == [
            (0, 'stream'), (6, 'of'), (8, 'different'),
            (17, 'sized'), (22, 'chunks')
        ]

    def test_stateful_behavior(self):
        """Test that DynamicIndexer maintains state across calls."""
        indexer = DynamicIndexer()
        assert indexer('a') == (0, 'a')
        assert indexer('b') == (1, 'b')
        assert indexer('c') == (2, 'c')


class TestDynamicallyIndex:
    """Tests for dynamically_index function."""

    def test_default_like_enumerate(self):
        """Test that default behavior matches enumerate."""
        stream = ['stream', 'of', 'different', 'sized', 'chunks']
        result = list(dynamically_index(stream, start=2))
        expected = list(enumerate(stream, start=2))
        assert result == expected

    def test_size_based_indexing(self):
        """Test dynamically_index with size increments."""
        stream = ['stream', 'of', 'different', 'sized', 'chunks']
        result = list(dynamically_index(stream, idx_updater=size_increments))
        assert result == [
            (0, 'stream'), (6, 'of'), (8, 'different'),
            (17, 'sized'), (22, 'chunks')
        ]


class TestSegmentOverlaps:
    """Tests for segment_overlaps function."""

    def test_overlapping_segments(self):
        """Test identifying overlapping segments."""
        overlaps = partial(segment_overlaps, query_bt=4, query_tt=8)

        test_segments = [
            (1, 3, 'completely before'),
            (2, 4, 'still completely before'),
            (3, 6, 'partially before, but overlaps bottom'),
            (4, 5, 'totally', 'inside'),
            (5, 8),
            (7, 10, 'partially after, but overlaps top'),
            (8, 11, 'completely after'),
            (100, 101, 'completely after (obviously)')
        ]

        result = list(filter(overlaps, test_segments))

        assert len(result) == 4
        assert (3, 6, 'partially before, but overlaps bottom') in result
        assert (4, 5, 'totally', 'inside') in result
        assert (5, 8) in result
        assert (7, 10, 'partially after, but overlaps top') in result

    def test_no_overlaps(self):
        """Test with segments that don't overlap."""
        overlaps = partial(segment_overlaps, query_bt=10, query_tt=20)
        segments = [(1, 5), (5, 9), (20, 25), (30, 35)]
        result = list(filter(overlaps, segments))
        assert result == []

    def test_exact_boundaries(self):
        """Test segment overlap with exact boundaries."""
        # Upper bound is strict, so (5, 10) should not overlap [10, 20)
        overlaps = partial(segment_overlaps, query_bt=10, query_tt=20)
        assert not segment_overlaps((5, 10), query_bt=10, query_tt=20)

        # But (5, 11) should overlap
        assert segment_overlaps((5, 11), query_bt=10, query_tt=20)


class TestBufferStats:
    """Tests for BufferStats class."""

    def test_basic_sum(self):
        """Test BufferStats with sum function."""
        bs = BufferStats(maxlen=4, func=sum)
        result = list(map(bs, range(7)))
        assert result == [0, 1, 3, 6, 10, 14, 18]

    def test_string_join(self):
        """Test BufferStats with string joining."""
        bs = BufferStats(maxlen=4, func=''.join)
        result = list(map(bs, 'abcdefgh'))
        assert result == ['a', 'ab', 'abc', 'abcd', 'bcde', 'cdef', 'defg', 'efgh']

    def test_product(self):
        """Test BufferStats with product function."""
        bs = BufferStats(maxlen=4, func=prod)
        result = list(map(bs, range(7)))
        assert result == [0, 0, 0, 0, 24, 120, 360]

    def test_appendleft(self):
        """Test BufferStats with appendleft."""
        bs = BufferStats(maxlen=4, func=''.join, add_new_val='appendleft')
        result = list(map(bs, 'abcdefgh'))
        assert result == ['a', 'ba', 'cba', 'dcba', 'edcb', 'fedc', 'gfed', 'hgfe']

    def test_stateful(self):
        """Test that BufferStats is stateful."""
        bs = BufferStats(maxlen=3, func=sum)
        assert bs(1) == 1
        assert bs(2) == 3
        assert bs(3) == 6
        assert bs(4) == 9  # buffer is now [2, 3, 4]
        assert bs(5) == 12  # buffer is now [3, 4, 5]

    def test_maxlen_required(self):
        """Test that maxlen is required."""
        with pytest.raises(TypeError, match='required to specify maxlen'):
            BufferStats(func=sum)

    def test_maxlen_must_be_int(self):
        """Test that maxlen must be an integer."""
        with pytest.raises(TypeError, match='maxlen must be an integer'):
            BufferStats(maxlen='not an int', func=sum)

    def test_custom_function(self):
        """Test BufferStats with custom function."""
        bs = BufferStats(maxlen=3, func=lambda buf: max(buf) - min(buf))
        result = [bs(x) for x in [5, 3, 8, 2, 9]]
        # [5] -> 0, [5,3] -> 2, [5,3,8] -> 5, [3,8,2] -> 6, [8,2,9] -> 7
        assert result == [0, 2, 5, 6, 7]


class TestReturnBufferOnStatsCondition:
    """Tests for return_buffer_on_stats_condition function."""

    def test_condition_true(self):
        """Test when condition is true."""
        result = return_buffer_on_stats_condition(
            stats=3,
            buffer=[1, 2, 3, 4],
            cond=lambda x: x % 2 == 1
        )
        assert result == [1, 2, 3, 4]

    def test_condition_false(self):
        """Test when condition is false."""
        result = return_buffer_on_stats_condition(
            stats=3,
            buffer=[1, 2, 3, 4],
            cond=lambda x: x % 2 == 0,
            else_val='3 is not even!'
        )
        assert result == '3 is not even!'

    def test_default_else_val(self):
        """Test default else_val of None."""
        result = return_buffer_on_stats_condition(
            stats=None,
            buffer=[1, 2, 3]
        )
        assert result is None


class TestSegmenter:
    """Tests for Segmenter class."""

    def test_basic_segmenter(self):
        """Test basic Segmenter usage."""
        bs = BufferStats(maxlen=10, func=sum)
        return_if_odd = partial(
            return_buffer_on_stats_condition,
            cond=lambda x: x % 2 == 1,
            else_val='The sum is not odd!'
        )
        seg = Segmenter(buffer=bs, stats_buffer_callback=return_if_odd)

        # Sum of [1] is 1 (odd)
        result = seg(new_val=1)
        assert result == [1]

        # Sum of [1, 2] is 3 (odd)
        result = seg(new_val=2)
        assert result == [1, 2]

        # Sum of [1, 2, 5] is 8 (even)
        result = seg(new_val=5)
        assert result == 'The sum is not odd!'

    def test_segmenter_accumulation(self):
        """Test that Segmenter accumulates values correctly."""
        bs = BufferStats(maxlen=5, func=sum)
        seg = Segmenter(
            buffer=bs,
            stats_buffer_callback=lambda stats, buf: (stats, list(buf))
        )

        result1 = seg(1)
        assert result1 == (1, [1])

        result2 = seg(2)
        assert result2 == (3, [1, 2])

        result3 = seg(3)
        assert result3 == (6, [1, 2, 3])

    def test_segmenter_with_buffer_overflow(self):
        """Test Segmenter when buffer overflows."""
        bs = BufferStats(maxlen=3, func=sum)
        seg = Segmenter(
            buffer=bs,
            stats_buffer_callback=lambda stats, buf: (stats, list(buf))
        )

        for i in range(1, 6):
            result = seg(i)

        # After adding 1,2,3,4,5 with maxlen=3, buffer should be [3,4,5]
        stats, buf = result
        assert buf == [3, 4, 5]
        assert stats == 12
