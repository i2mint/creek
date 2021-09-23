"""The base objects of creek"""

from creek.util import cls_wrap, static_identity_method, no_such_item


class Creek:
    """A layer-able version of the stream interface

    There are three layering methods -- `pre_iter`, `data_to_obj`, and `post_iter`
    -- whose use is demonstrated in the iteration code below:

    >>> from io import StringIO
    >>>
    >>> src = StringIO(
    ... '''a, b, c
    ... 1,2, 3
    ... 4, 5,6
    ... '''
    ... )
    >>>
    >>> from creek.base import Creek
    >>>
    >>> class MyCreek(Creek):
    ...     def data_to_obj(self, line):
    ...         return [x.strip() for x in line.strip().split(',')]
    ...
    >>> stream = MyCreek(src)
    >>>
    >>> list(stream)
    [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]

    If we try that again, we'll get an empty list since the cursor is at the end.

    >>> list(stream)
    []

    But if the underlying stream has a seek, so does the creek, so we can "rewind"

    >>> stream.seek(0)
    0

    >>> list(stream)
    [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]

    You can also use ``next`` to get stream items one by one

    >>> stream.seek(0)  # rewind again to get back to the beginning
    0
    >>> next(stream)
    ['a', 'b', 'c']
    >>> next(stream)
    ['1', '2', '3']

    Let's add a filter! There's two kinds you can use.
    One that is applied to the line before the data is transformed by data_to_obj,
    and the other that is applied after (to the obj).

    >>> from creek.base import Creek
    >>> from io import StringIO
    >>>
    >>> src = StringIO(
    ...     '''a, b, c
    ... 1,2, 3
    ... 4, 5,6
    ... ''')
    >>> class MyFilteredCreek(MyCreek):
    ...     def post_iter(self, objs):
    ...         yield from filter(lambda obj: str.isnumeric(obj[0]), objs)
    >>>
    >>> s = MyFilteredCreek(src)
    >>>

    >>> list(s)
    [['1', '2', '3'], ['4', '5', '6']]
    >>> s.seek(0)
    0
    >>> next(s)
    ['1', '2', '3']
    >>> next(s)
    ['4', '5', '6']

    Recipes:

    - `pre_iter`: involving `itertools.islice` to skip header lines
    - `pre_iter`: involving enumerate to get line indices in stream iterator
    - `pre_iter = functools.partial(map, pre_proc_func)` to preprocess all streamitems \
        with `pre_proc_func`
    - `pre_iter`: include filter before obj
    - `post_iter`: `chain.from_iterable` to flatten a chunked/segmented stream
    - `post_iter`: `functools.partial(filter, condition)` to filter yielded objs

    """

    def __init__(self, stream):
        self.stream = stream

    wrap = classmethod(cls_wrap)

    def __getattr__(self, attr):
        """Delegate method to wrapped stream"""
        return getattr(self.stream, attr)

    def __dir__(self):
        return list(
            set((*dir(self.__class__), *dir(self.stream)))
        )  # to forward dir to delegated stream as well
        # return list(set(dir(self.__class__)).union(self.stream.__dir__()))  # to forward dir to delegated stream as well

    def __hash__(self):
        return self.stream.__hash__()

    # _data_of_obj = static_identity_method  # for write methods
    pre_iter = static_identity_method
    data_to_obj = static_identity_method
    # post_filt = stream_util.always_true
    post_iter = static_identity_method

    def __iter__(self):
        yield from self.post_iter(map(self.data_to_obj, self.pre_iter(self.stream)))

        # for line in self.pre_iter(self.stream):
        #     obj = self.data_to_obj(line)
        #     if self.post_filt(obj):
        #         yield obj

        # TODO: See pros and cons of above vs below:
        # yield from filter(self.post_filt,
        #                   map(self.data_to_obj,
        #                       self.pre_iter(self.stream)))

    # _wrapped_methods = {'__iter__'}

    def __next__(self):  # TODO: Pros and cons of having a __next__?
        """by default: next(iter(self))
        Expect to
        """
        return next(iter(self))

    def __enter__(self):
        self.stream.__enter__()
        return self
        # return self._pre_proc(self.stream) # moved to iter to

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.stream.__exit__(
            exc_type, exc_val, exc_tb
        )  # TODO: Should we have a _post_proc? Uses?


# class Brook(Creek):
#     post_iter = static_identity_method
#
#     def __iter__(self):
#         yield from self.post_iter(
#             filter(self.post_filt,
#                    map(self.data_to_obj,
#                        self.pre_iter(
#                            self.stream))))
