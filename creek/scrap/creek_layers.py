"""
# Wrapper interfaces

## Inner-class

```python
def intify(self, data):
    return tuple(map(int, data))

class D(Creek):
    # subclassing `CreekLayer` indicates that this class is a layering class
    # could also use decorator for this: Allowing simple injection of external classes
    class Lay(CreekLayer):
        # name indicates what kind of layer this is (i.e. where/how to apply it)
        def pre_iter(stream):
            next(stream)  # skip one

        @data_to_obj  # decorator to indicate what kind of layer this is (i.e. where/how to apply it
        def strip_and_split(data):  # function can be a method (first arg is instance) or static (data_to_obj figures it out)
            return data.strip().split(',')

        another_data_to_obj_layer = data_to_obj(intify)  # decorator can be used to inject function defined externally
```

## Decorators

```python
@lay(kind='pre_iter', func=func)
@lay.data_to_obj(func=func)
class D(Creek):
    pass
```

## Fluid interfaces

```python
D = (Creek
    .lay('pre_iter', func)
    .lay.data_to_obj(func)...
)
```


# Backend

Use lists to stack layers.

Compile the layers to increase resource use.

Uncompile to increase debugibility.
"""

from typing import Iterable, Callable
from dataclasses import dataclass
from inspect import Signature, signature


def identity_func(x):
    return x


static_identity_method = staticmethod(identity_func)


class Compose:
    def __init__(self, *funcs, default=identity_func):
        if len(funcs) == 0:
            self.first_func = (default,)
            self.other_funcs = ()
        else:
            self.first_func, *self.other_funcs = funcs
        # The following so that __call__ gets the same signature as first_func:
        self.__signature__ = Signature(
            list(signature(self.first_func).parameters.values())
        )

    def __call__(self, *args, **kwargs):
        out = self.first_func(*args, **kwargs)
        for func in self.other_funcs:
            out = func(out)
        return out


FuncSequence = Iterable[Callable]


# TODO: Use descriptors to manage the pre_iters/pre_iter relationship.
@dataclass
class CreekLayer:
    pre_iters: FuncSequence = ()
    data_to_objs: FuncSequence = ()
    post_iters: FuncSequence = ()

    def pre_iter(self, data_stream):
        return Compose(*self.pre_iters)(data_stream)

    def data_to_obj(self, data):
        return Compose(*self.data_to_objs)(data)

    def post_iter(self, obj_stream):
        return Compose(*self.post_iters)(obj_stream)

    def lay(self, **kwargs):
        pass
