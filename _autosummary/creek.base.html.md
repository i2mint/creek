# creek.base

The `Creek` base class: a layer-able wrapper around a stream.

Main entry points:

- `Creek`: subclass it and override `pre_iter`, `data_to_obj` and/or `post_iter`
- `Creek.wrap`: wrap a stream class (or instance) so that you get creeks out of it

```pycon
>>> from creek.base import Creek
>>> class Doubler(Creek):
...     def data_to_obj(self, x):
...         return x * 2
>>> list(Doubler([1, 2, 3]))
[2, 4, 6]
```

### Classes

| [`Creek`](#creek.base.Creek)(stream)   | A layer-able version of the stream interface.   |
|------------------------------------------------------------------|-------------------------------------------------|

### *class* creek.base.Creek(stream)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A layer-able version of the stream interface.

A `Creek` wraps a stream (any iterable, typically a file-like object) and
delegates attribute access to it, so it can be used where the stream was.
Iteration goes through three layering methods, each the identity by default:

- `pre_iter(stream)`: prepare and/or filter the raw stream
- `data_to_obj(item)`: transform each item the stream yields
- `post_iter(objs)`: further process or filter the transformed objects

That is, `iter(creek)` is `post_iter(map(data_to_obj, pre_iter(stream)))`.

* **Parameters:**
  **stream** – The wrapped stream. Attributes not defined on the creek
  (`seek`, `close`…) are looked up on it.

```pycon
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
```

If we try that again, we’ll get an empty list since the cursor is at the end.

```pycon
>>> list(stream)
[]
```

But if the underlying stream has a seek, so does the creek, so we can “rewind”

```pycon
>>> stream.seek(0)
0
```

```pycon
>>> list(stream)
[['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]
```

You can also use `next` to get stream items one by one

```pycon
>>> stream.seek(0)  # rewind again to get back to the beginning
0
>>> next(stream)
['a', 'b', 'c']
>>> next(stream)
['1', '2', '3']
```

Let’s add a filter! There’s two kinds you can use.
One that is applied to the line before the data is transformed by data_to_obj,
and the other that is applied after (to the obj).

```pycon
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
```

```pycon
>>> list(s)
[['1', '2', '3'], ['4', '5', '6']]
>>> s.seek(0)
0
>>> next(s)
['1', '2', '3']
>>> next(s)
['4', '5', '6']
```

Recipes:

- `pre_iter`: involving `itertools.islice` to skip header lines
- `pre_iter`: involving enumerate to get line indices in stream iterator
- `pre_iter = functools.partial(map, pre_proc_func)` to preprocess all stream
  items with `pre_proc_func`
- `pre_iter`: include filter before obj
- `post_iter`: `chain.from_iterable` to flatten a chunked/segmented stream
- `post_iter`: `functools.partial(filter, condition)` to filter yielded objs

#### *static* data_to_obj(x)

Return `x` unchanged.

#### *static* post_iter(x)

Return `x` unchanged.

#### *static* pre_iter(x)

Return `x` unchanged.

#### *classmethod* wrap(obj)

Wrap `obj` in `cls`: a subclass wrapping instances at construction if `obj` is a class, else `cls(obj)`.
