# creek.multi_streams

Merge several sorted streams into one stream of `(stream_id, item)` pairs.

Main entry points:

- `MergedStreams`: the merge, driven by `heapq.merge` on a sort key of the items
- `multi_stream_items`: tag each stream’s items with its id, one iterable per stream

```pycon
>>> from creek.multi_streams import MergedStreams
>>> list(MergedStreams({'a': [1, 3], 'b': [2]}))
[('a', 1), ('b', 2), ('a', 3)]
```

### Functions

| [`multi_stream_items`](#creek.multi_streams.multi_stream_items)(streams_map)        | Provides a iterable of (k1, v1_1), (k1, v1_2), ...   |
|-----------------------------------------------------------------------------------------|------------------------------------------------------|
| [`transform_methods`](#creek.multi_streams.transform_methods)(cls[, method_trans]) | Applies method_trans to all the methods of `cls`     |

### Classes

| [`MergedStreams`](#creek.multi_streams.MergedStreams)(streams_map[, sort_key])   | Creates an iterable of `(stream_id, stream_item)` pairs from a stream Mapping, that is, `{stream_id: stream, ...}`: A sort of "flattening" of the Mapping.   |
|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`SortKeys`](#creek.multi_streams.SortKeys)()                               | A few `itemgetter` sort keys, exposed as staticmethods.                                                                                                      |

### *class* creek.multi_streams.MergedStreams(streams_map, sort_key=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Creates an iterable of `(stream_id, stream_item)` pairs from a stream Mapping,
that is, `{stream_id: stream, ...}`: A sort of “flattening” of the Mapping.

This can be useful, for instance, if you want to make “slabs” of data, gathering
together all the data for a given time period, from multiple streams.

The `stream_item` will be yielded in sorted order.
Sort behavior can be modified by the `sort_key` argument which behaves like `key`
arguments of built-in like `sorted`, `heapq.merge`, `itertools.groupby`, etc.

If given, the `sort_key` function applies to `stream_item` (not to `stream_id`).

#### IMPORTANT
To function as expected, the streams should be already sorted (according
to the `sort_key` order).

The cannonical use case of this function is to “flatten”, or “weave together”
multiple streams of timestamped data. We’re given several streams that provide
`(timestamp, data)` items (where timestamps arrive in order within each stream)
and we get a single stream of `(stream_id, (timestamp, data))` items where
the `timestamp` values are yielded in sorted order.

The following example uses a dict pointing to a fixed-size list as the `stream_map`
but in general the `stream_map` will be a `Mapping` (not necessarily a dict)
whose values are potentially bound-less streams.

```pycon
>>> streams_map = {
...     'hello': [(2, 'two'), (3, 'three'), (5, 'five')],
...     'world': [(0, 'zero'), (1, 'one'), (3, 'three'), (6, 'six')]
... }
>>> streams_items = MergedStreams(streams_map)
>>> it = iter(streams_items)
>>> list(it)
[('world', (0, 'zero')),
 ('world', (1, 'one')),
 ('hello', (2, 'two')),
 ('hello', (3, 'three')),
 ('world', (3, 'three')),
 ('hello', (5, 'five')),
 ('world', (6, 'six'))]
```

### *class* creek.multi_streams.SortKeys

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A few `itemgetter` sort keys, exposed as staticmethods.

### creek.multi_streams.multi_stream_items(streams_map)

Provides a iterable of (k1, v1_1), (k1, v1_2), …

```pycon
>>> streams_map = {'hello': 'abc', 'world': [1, 2]}
>>> hello_items, world_items = multi_stream_items(streams_map)
>>> list(hello_items)
[('hello', 'a'), ('hello', 'b'), ('hello', 'c')]
>>> list(world_items)
[('world', 1), ('world', 2)]
```

### creek.multi_streams.staticmethods(cls, \*, method_trans=<class 'staticmethod'>)

Applies method_trans to all the methods of `cls`

```pycon
>>> from functools import partial
>>> staticmethods = partial(transform_methods, method_trans=staticmethod)
```

Now staticmethods is a class decorator that can be used to make all methods
be defined as staticmethods in bulk

```pycon
>>> @staticmethods
... class C:
...     foo = lambda x: x + 1
...     bar = lambda y: y * 2
>>> c = C()
>>> c.foo(6)
7
>>> c.bar(6)
12
```

### creek.multi_streams.transform_methods(cls, method_trans=<class 'staticmethod'>)

Applies method_trans to all the methods of `cls`

```pycon
>>> from functools import partial
>>> staticmethods = partial(transform_methods, method_trans=staticmethod)
```

Now staticmethods is a class decorator that can be used to make all methods
be defined as staticmethods in bulk

```pycon
>>> @staticmethods
... class C:
...     foo = lambda x: x + 1
...     bar = lambda y: y * 2
>>> c = C()
>>> c.foo(6)
7
>>> c.bar(6)
12
```
