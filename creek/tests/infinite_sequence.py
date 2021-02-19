import pytest
from creek.infinite_sequence import InfiniteSeq, IndexedBuffer, OverlapsFutureError, OverlapsPastError


def test_infinite_seq():
    from itertools import chain
    data_gen_source = [
        range(0, 5),
        range(5, 12),
        range(12, 22),
        range(22, 29),
    ]

    data_gen = lambda: chain.from_iterable(data_gen_source)
    assert list(data_gen()) == list(range(29))

    s = InfiniteSeq(iterator=data_gen(), buffer_len=5)
    assert s[2:6] == [2, 3, 4, 5]
    assert s[1] == 1  # still in the buffer
    assert s[10:15] == [10, 11, 12, 13, 14]
    assert s[23:24] == [23]


def test_indexed_buffer_common_case():
    # from creek.scrap.infinite_sequence import InfiniteSeq
    s = IndexedBuffer(maxlen=4)
    s.extend(range(4))
    assert list(s) == [0, 1, 2, 3]
    assert s[2] == 2
    assert s[1:2] == [1]
    assert s[1:1] == []
    s.append(4)
    s.append(5)
    assert list(s) == [2, 3, 4, 5]
    assert s[2] == 2
    assert s[5] == 5
    assert s[2:5] == [2, 3, 4]
    assert s[3:6] == [3, 4, 5]
    assert s[2:6] == list(range(2, 6))
    with pytest.raises(OverlapsPastError) as excinfo:
        s[1:4]  # element for idx 1 is missing in [2, 3, 4, 5]
        assert "in the past" in excinfo.value
    with pytest.raises(OverlapsPastError) as excinfo:
        s[0:9]  # elements for 0:2 are missing (as well as 6:9, but OverlapsPastError trumps OverlapsFutureError
    with pytest.raises(OverlapsFutureError) as excinfo:
        s[4:9]  # element for 6:9 are missing in [2, 3, 4, 5]
        assert "in the future" in excinfo.value


# simple_test()

def test_indexed_buffer_extreme_cases():
    s = IndexedBuffer(maxlen=7)
    # when there's nothing and you ask for something
    with pytest.raises(OverlapsFutureError):
        s[0]
    with pytest.raises(OverlapsFutureError):
        s[:3]

    # when there's something, but buffer is not full, but you ask for something that hasn't happened yet.
    s.extend(range(5))  # buffer now has the 0:5 view (but is not full!)
    assert list(s) == [0, 1, 2, 3, 4]  # this is what's in the buffer now
    with pytest.raises(OverlapsFutureError) as excinfo:
        s[10:14]  # completely in the future (0:5 "happens before" 10:14 (Allen's interval algebra terminology))
        assert "in the future" in excinfo.value
    with pytest.raises(OverlapsFutureError) as excinfo:
        s[3:7]  # overlaps with 0:5
        assert "in the future" in excinfo.value

    s.extend(range(5, 10))  # add more data (making the buffer full and shifted)
    assert list(s) == [3, 4, 5, 6, 7, 8, 9]  # this is what's in the buffer now
    assert s[3:9:2] == [3, 5, 7], "slices with steps don't work"

    # use negative indices
    assert s[4:-1] == [4, 5, 6, 7, 8]
    assert s[-4:-1] == [6, 7, 8]

