import functools

wrapper_assignments = (
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__annotations__",
    "__defaults__",
    "__kwdefaults__",
)

update_wrapper = functools.update_wrapper
update_wrapper.__defaults__ = (
    functools.WRAPPER_ASSIGNMENTS,
    functools.WRAPPER_UPDATES,
)
wraps = functools.wraps
wraps.__defaults__ = (functools.WRAPPER_ASSIGNMENTS, functools.WRAPPER_UPDATES)


# TODO: Make identity_func "identifiable". If we use the following one, we can use == to detect it's use,
# TODO: ... but there may be a way to annotate, register, or type any identity function so it can be detected.
def identity_func(x):
    return x


static_identity_method = staticmethod(identity_func)


class NoSuchItem:
    pass


no_such_item = NoSuchItem()


def cls_wrap(cls, obj):
    if isinstance(obj, type):

        @wraps(obj, updated=())
        class Wrap(cls):
            @wraps(obj.__init__)
            def __init__(self, *args, **kwargs):
                wrapped = obj(*args, **kwargs)
                super().__init__(wrapped)

        # Wrap.__signature__ = signature(obj)

        return Wrap
    else:
        return cls(obj)


class stream_util:
    def always_true(*args, **kwargs):
        return True

    def do_nothing(*args, **kwargs):
        pass

    def rewind(self, instance):
        instance.seek(0)

    def skip_lines(self, instance, n_lines_to_skip=0):
        instance.seek(0)


class Creek:
    """A layer-able version of the stream interface

    There are three layering methods -- pre_iter, data_to_obj, and post_iter
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
    >>> stream.seek(0)  # oh!... but we consumed the stream already, so let's go back to the beginning
    0
    >>> list(stream)
    [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]
    >>> stream.seek(0)  # reverse again
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
    >>> list(s)
    [['1', '2', '3'], ['4', '5', '6']]
    >>> s.seek(0)
    0
    >>> next(s)
    ['1', '2', '3']

    Recipes:
    - pre_iter: involving itertools.islice to skip header lines
    - pre_iter: involving enumerate to get line indices in stream iterator
    - pre_iter = functools.partial(map, line_pre_proc_func) to preprocess all lines with line_pre_proc_func
    - pre_iter: include filter before obj

    - post_iter: chain.from_iterable to flatten a chunked/segmented stream
    - post_iter: functools.partial(filter, condition) to filter yielded objs
    """

    def __init__(self, stream):
        self.stream = stream

    wrap = classmethod(cls_wrap)

    def __getattr__(self, attr):
        """Delegate method to wrapped stream"""
        return getattr(self.stream, attr)

    def __dir__(self):
        return list(set(dir(self.__class__)).union(self.stream.__dir__()))  # to forward dir to delegated stream as well

    def __hash__(self):
        return self.stream.__hash__()

    # _data_of_obj = static_identity_method  # for write methods
    pre_iter = static_identity_method
    data_to_obj = static_identity_method
    # post_filt = stream_util.always_true
    post_iter = static_identity_method

    def __iter__(self):
        yield from self.post_iter(
            map(self.data_to_obj,
                self.pre_iter(
                    self.stream)))

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
