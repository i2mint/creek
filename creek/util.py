"""Utils for creek"""

from functools import (
    WRAPPER_ASSIGNMENTS,
    partial,
    update_wrapper as _update_wrapper,
    wraps as _wraps,
)
from itertools import islice

wrapper_assignments = (*WRAPPER_ASSIGNMENTS, '__defaults__', '__kwdefaults__')
update_wrapper = partial(_update_wrapper, assigned=wrapper_assignments)
wraps = partial(_wraps, assigned=wrapper_assignments)


class NoSuchItem:
    pass


no_such_item = NoSuchItem()


class stream_util:
    def always_true(*args, **kwargs):
        return True

    def do_nothing(*args, **kwargs):
        pass

    def rewind(self, instance):
        instance.seek(0)

    def skip_lines(self, instance, n_lines_to_skip=0):
        instance.seek(0)


class PreIter:
    def skip_items(self, instance, n):
        return islice(instance, n, None)


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


# TODO: Make identity_func "identifiable". If we use the following one, we can use == to detect it's use,
# TODO: ... but there may be a way to annotate, register, or type any identity function so it can be detected.


def identity_func(x):
    return x


static_identity_method = staticmethod(identity_func)
