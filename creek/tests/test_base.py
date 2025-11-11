"""Unit tests for Creek base class."""
import pytest
from io import StringIO
from creek import Creek


class MyCreek(Creek):
    """Test Creek subclass that parses CSV-like data."""

    def data_to_obj(self, line):
        return [x.strip() for x in line.strip().split(',')]


class MyFilteredCreek(MyCreek):
    """Test Creek subclass with filtering."""

    def post_iter(self, objs):
        yield from filter(lambda obj: str.isnumeric(obj[0]), objs)


def test_creek_basic_iteration():
    """Test basic Creek iteration with data transformation."""
    src = StringIO(
        '''a, b, c
1,2, 3
4, 5,6
'''
    )

    stream = MyCreek(src)
    result = list(stream)

    assert result == [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]


def test_creek_seek():
    """Test Creek seek functionality."""
    src = StringIO(
        '''a, b, c
1,2, 3
4, 5,6
'''
    )

    stream = MyCreek(src)

    # Consume the stream
    list(stream)

    # Seek back to beginning
    stream.seek(0)

    # Should be able to iterate again
    result = list(stream)
    assert result == [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]


def test_creek_next():
    """Test Creek __next__ method."""
    src = StringIO(
        '''a, b, c
1,2, 3
4, 5,6
'''
    )

    stream = MyCreek(src)

    assert next(stream) == ['a', 'b', 'c']
    assert next(stream) == ['1', '2', '3']
    assert next(stream) == ['4', '5', '6']

    # Should raise StopIteration when exhausted
    with pytest.raises(StopIteration):
        next(stream)


def test_creek_with_filter():
    """Test Creek with post_iter filtering."""
    src = StringIO(
        '''a, b, c
1,2, 3
4, 5,6
'''
    )

    stream = MyFilteredCreek(src)
    result = list(stream)

    # Should only get rows starting with numbers
    assert result == [['1', '2', '3'], ['4', '5', '6']]


def test_creek_filter_seek():
    """Test Creek filtering with seek."""
    src = StringIO(
        '''a, b, c
1,2, 3
4, 5,6
'''
    )

    stream = MyFilteredCreek(src)

    # First iteration
    result1 = list(stream)
    assert result1 == [['1', '2', '3'], ['4', '5', '6']]

    # Seek and iterate again
    stream.seek(0)
    result2 = list(stream)
    assert result2 == [['1', '2', '3'], ['4', '5', '6']]


def test_creek_filter_next():
    """Test Creek filtering with next."""
    src = StringIO(
        '''a, b, c
1,2, 3
4, 5,6
'''
    )

    stream = MyFilteredCreek(src)

    assert next(stream) == ['1', '2', '3']
    assert next(stream) == ['4', '5', '6']


def test_creek_context_manager():
    """Test Creek as context manager."""
    src = StringIO(
        '''a, b, c
1,2, 3
'''
    )

    with MyCreek(src) as stream:
        result = list(stream)
        assert result == [['a', 'b', 'c'], ['1', '2', '3']]

    # Stream should be closed after context
    assert src.closed


def test_creek_delegates_to_stream():
    """Test that Creek delegates unknown attributes to wrapped stream."""
    src = StringIO('test')
    stream = Creek(src)

    # Should be able to access StringIO methods
    assert hasattr(stream, 'read')
    assert hasattr(stream, 'seek')
    assert hasattr(stream, 'tell')


def test_creek_identity_methods():
    """Test default identity behavior of Creek."""
    src = StringIO('line1\nline2\nline3\n')
    stream = Creek(src)

    # Default behavior: no transformation
    result = list(stream)
    assert result == ['line1\n', 'line2\n', 'line3\n']


class PreIterCreek(Creek):
    """Test Creek with pre_iter modification."""

    def pre_iter(self, stream):
        # Skip first line (header)
        next(stream)
        return stream


def test_creek_pre_iter():
    """Test Creek with pre_iter to skip header."""
    src = StringIO('header\nline1\nline2\n')
    stream = PreIterCreek(src)

    result = list(stream)
    assert result == ['line1\n', 'line2\n']
    assert 'header' not in str(result)


class TransformCreek(Creek):
    """Test Creek with all three transformation methods."""

    def pre_iter(self, stream):
        # Enumerate lines
        return enumerate(stream)

    def data_to_obj(self, item):
        # Extract line number and strip line
        idx, line = item
        return {'index': idx, 'line': line.strip()}

    def post_iter(self, objs):
        # Filter out empty lines
        yield from filter(lambda obj: obj['line'], objs)


def test_creek_all_transformations():
    """Test Creek with all three transformation layers."""
    src = StringIO('line1\n\nline3\n')
    stream = TransformCreek(src)

    result = list(stream)

    # Should have indexed, transformed, and filtered
    assert result == [
        {'index': 0, 'line': 'line1'},
        {'index': 2, 'line': 'line3'}
    ]
    # Empty line should be filtered out
    assert len(result) == 2
