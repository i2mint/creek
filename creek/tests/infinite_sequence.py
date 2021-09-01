import pytest
from creek.infinite_sequence import (
    InfiniteSeq,
    IndexedBuffer,
    OverlapsFutureError,
    OverlapsPastError,
)


def test_infinite_seq():
    from itertools import cycle

    iterator = cycle(range(100))
    # Let's make an InfiniteSeq instance for this stream, accomodating for a view of up to 11 items.
    s = InfiniteSeq(iterator, buffer_len=11)
    # Let's ask for element 15 (which is the (15 + 1)th element (and should have a value of 15).
    assert s[15] == 15
    # Now, to get this value, the iterator will move forward up to that point;
    # that is, until the buffer's head (i.e. most recent) item contains that requested (15 + 1)th element.
    # But the buffer is of size 11, so we still have access to a few previous elements:
    assert s[11] == 11
    assert s[5:15] == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # But if we asked for anything before index 5...
    with pytest.raises(OverlapsPastError):
        _ = s[2:7]

    # So we can't go backwards. But we can always go forwards:

    assert s[95:105] == [95, 96, 97, 98, 99, 0, 1, 2, 3, 4]

    # You can also use slices with step and with negative integers (referencing the head of the buffer)
    assert s[120:130:2] == [20, 22, 24, 26, 28]
    assert s[-8:-2] == [22, 23, 24, 25, 26, 27]

    # What to do if your iterator provides "chunks"? Example below.
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
        assert 'in the past' in excinfo.value
    with pytest.raises(OverlapsPastError) as excinfo:
        s[
            0:9
        ]  # elements for 0:2 are missing (as well as 6:9, but OverlapsPastError trumps OverlapsFutureError
    with pytest.raises(OverlapsFutureError) as excinfo:
        s[4:9]  # element for 6:9 are missing in [2, 3, 4, 5]
        assert 'in the future' in excinfo.value


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
        s[
            10:14
        ]  # completely in the future (0:5 "happens before" 10:14 (Allen's interval algebra terminology))
        assert 'in the future' in excinfo.value
    with pytest.raises(OverlapsFutureError) as excinfo:
        s[3:7]  # overlaps with 0:5
        assert 'in the future' in excinfo.value

    s.extend(range(5, 10))  # add more data (making the buffer full and shifted)
    assert list(s) == [3, 4, 5, 6, 7, 8, 9]  # this is what's in the buffer now
    assert s[3:9:2] == [3, 5, 7], "slices with steps don't work"

    # use negative indices
    assert s[4:-1] == [4, 5, 6, 7, 8]
    assert s[-4:-1] == [6, 7, 8]


def test_source(capsys):
    from creek.infinite_sequence import InfiniteSeq
    from collections import Mapping

    def assert_prints(print_str):
        out, err = capsys.readouterr()
        assert out == print_str

    class Source(Mapping):
        n = 100

        __len__ = lambda self: self.n

        def __iter__(self):
            yield from range(self.n)

        def __getitem__(self, k):
            print(f'Asking for {k}')
            return list(range(k * 10, (k + 1) * 10))

    source = Source()

    assert source[3] == [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    assert_prints('Asking for 3\n')

    from itertools import chain

    iterator = chain.from_iterable(source.values())

    s = InfiniteSeq(iterator, 10)

    assert s[:5] == [0, 1, 2, 3, 4]
    assert_prints('Asking for 0\n')

    assert s[4:8] == [4, 5, 6, 7]

    assert s[8:12] == [8, 9, 10, 11]
    assert_prints('Asking for 1\n')

    assert s[40:42] == [40, 41]
    assert_prints('Asking for 2\nAsking for 3\nAsking for 4\n')
