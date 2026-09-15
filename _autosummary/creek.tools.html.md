# creek.tools

Tools to index, window and segment streams.

Everything here is a plain function or a small callable class that you compose
with `map`, `filter` and `functools.partial`; nothing depends on `Creek`.

Main entry points:

- `filter_and_index_stream`: index a stream (`enumerate` by default) and filter on the items
- `dynamically_index / DynamicIndexer`: `enumerate` where you choose how the index advances
- `BufferStats`: a fixed-size buffer that returns a statistic of its contents each time it is fed
- `Segmenter`: feed a `BufferStats` and hand its contents to a callback after each value

```pycon
>>> from creek.tools import BufferStats
>>> list(map(BufferStats(maxlen=3, func=sum), range(5)))
[0, 1, 3, 6, 9]
```

### Functions

| [`alt_dynamically_index`](#creek.tools.alt_dynamically_index)([idx_updater, start])       | Alternative to dynamically_index using itertools and partial                                                                                                                                                                          |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`always_true`](#creek.tools.always_true)(x)                                    | Return `True`, whatever `x` is.                                                                                                                                                                                                       |
| [`apply_and_fanout`](#creek.tools.apply_and_fanout)(seq, func, idx)                  | Apply function (that returns an Iterable) to an element of a sequence and fanout (broadcast) the resulting items, to produce an iterable of tuples each containing one of these items along with 'a copy' of the other tuple elements |
| [`apply_func_to_index`](#creek.tools.apply_func_to_index)(seq, apply_to_idx, func)      | Apply `func` to the element at `apply_to_idx` of `seq`, returning a tuple.                                                                                                                                                            |
| [`count_increments`](#creek.tools.count_increments)(current_idx, obj[, step])        | Index updater that ignores `obj` and advances the index by `step`.                                                                                                                                                                    |
| [`current_time`](#creek.tools.current_time)(current_idx, obj)                    | Doesn't even look at current_idx or obj.                                                                                                                                                                                              |
| [`dynamically_index`](#creek.tools.dynamically_index)(iterable[, start, idx_updater]) | Generalization of `enumerate(iterable)` that allows one to specify how the indices should be updated.                                                                                                                                 |
| [`fanout_and_flatten`](#creek.tools.fanout_and_flatten)(iterable_of_seqs, func, idx)   | Apply apply_and_fanout to an iterable of sequences.                                                                                                                                                                                   |
| [`fanout_and_flatten_dicts`](#creek.tools.fanout_and_flatten_dicts)(iterable_of_dicts, ...)  | Apply apply_and_fanout to an iterable of dicts.                                                                                                                                                                                       |
| [`filter_and_index_stream`](#creek.tools.filter_and_index_stream)(stream, data_item_filt)   | Index a stream and filter it (based only on the data items).                                                                                                                                                                          |
| [`is_not_none`](#creek.tools.is_not_none)(x)                                    | Return `True` if `x` is not `None`.                                                                                                                                                                                                   |
| [`return_buffer_on_stats_condition`](#creek.tools.return_buffer_on_stats_condition)(stats, buffer)   | Return `buffer` if `cond(stats)` holds, else `else_val`.                                                                                                                                                                              |
| [`segment_overlaps`](#creek.tools.segment_overlaps)(bt_tt_segment, query_bt, ...)    | Returns True if, and only if, bt_tt_segment overlaps query interval.                                                                                                                                                                  |
| [`size_increments`](#creek.tools.size_increments)(current_idx, obj[, size_func])    | Index updater that advances the index by `size_func(obj)`, `len` by default.                                                                                                                                                          |

### Classes

| [`BufferStats`](#creek.tools.BufferStats)([values, maxlen, func, add_new_val])   | A callable, fixed-size (fifo) buffer returning a statistic of its contents on every call.   |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`DynamicIndexer`](#creek.tools.DynamicIndexer)([start, idx_updater])               | Turn data items into `(index, item)` pairs, with a pluggable index update rule.             |
| [`Segmenter`](#creek.tools.Segmenter)(buffer[, stats_buffer_callback])         | Feed values to a `BufferStats` and hand its contents to a callback after each one.          |

### *class* creek.tools.BufferStats(values=(), maxlen=<object object>, func=<built-in function sum>, add_new_val=<method 'append' of 'collections.deque' objects>)

Bases: [`deque`](https://docs.python.org/3/library/collections.html#collections.deque)

A callable, fixed-size (fifo) buffer returning a statistic of its contents on every call.

Each call adds the input to the buffer (`deque.append` by default, configurable
through `add_new_val`) and returns `func` applied to the buffer’s contents.
Use it with `map` to get rolling-window statistics of a stream.

* **Parameters:**
  * **values** – Initial contents of the buffer.
  * **maxlen** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Size of the buffer. Required.
  * **func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – The function computed on the buffer contents and returned on each
    call; `sum` by default.
  * **add_new_val** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – The `(buffer, new_val)` function that adds a value to the
    buffer: a `deque` method (`deque.append` by default, `deque.extend`,
    `deque.appendleft`…), its name as a string, or any function with that
    signature.
* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – If `maxlen` is not given, or is not an `int`.

```pycon
>>> bs = BufferStats(maxlen=4, func=sum)
>>> list(map(bs, range(7)))
[0, 1, 3, 6, 10, 14, 18]
```

See what happens when you feed the same sequence again:

```pycon
>>> list(map(bs, range(7)))
[15, 12, 9, 6, 10, 14, 18]
```

More examples:

```pycon
>>> list(map(BufferStats(maxlen=4, func=''.join), 'abcdefgh'))
['a', 'ab', 'abc', 'abcd', 'bcde', 'cdef', 'defg', 'efgh']
```

```pycon
>>> from math import prod
>>> list(map(BufferStats(maxlen=4, func=prod), range(7)))
[0, 0, 0, 0, 24, 120, 360]
```

With a different `add_new_val` choice.

```pycon
>>> bs = BufferStats(maxlen=4, func=''.join, add_new_val=deque.appendleft)
>>> list(map(bs, 'abcdefgh'))
['a', 'ba', 'cba', 'dcba', 'edcb', 'fedc', 'gfed', 'hgfe']
```

With `add_new_val=deque.extend`, data can be fed in chunks.
In the following, also see how we use iterize to get a function that
takes an iterator and returns an iterator

```pycon
>>> from creek.util import iterize
>>> window_stats = iterize(BufferStats(
...     maxlen=4, func=''.join, add_new_val=deque.extend)
... )
>>> chks = ['a', 'bc', 'def', 'gh']
>>> for x in window_stats(chks):
...     print(x)
a
abc
cdef
efgh
```

#### NOTE
To those who might think that they can optimize this for special
cases: Yes you can.
But SHOULD you? Is it worth the increase in complexity and reduction in
flexibility?
See [https://github.com/thorwhalen/umpyre/blob/master/misc/performance_of_rolling_window_stats.md](https://github.com/thorwhalen/umpyre/blob/master/misc/performance_of_rolling_window_stats.md)

### *class* creek.tools.DynamicIndexer(start=0, idx_updater=<function count_increments>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Turn data items into `(index, item)` pairs, with a pluggable index update rule.

* **Parameters:**
  * **start** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The index to start at (the first data item will have this index)
  * **idx_updater** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`DataItem`)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The `(current_idx, data_item) -> next_idx` function
    computing the index the next data item will get

Let’s take a finite stream of finite iterables (strings here):

```pycon
>>> stream = ['stream', 'of', 'different', 'sized', 'chunks']
```

The default `DynamicIndexer` just does what `enumerate` does:

```pycon
>>> counter_index = DynamicIndexer()
>>> list(map(counter_index, stream))
[(0, 'stream'), (1, 'of'), (2, 'different'), (3, 'sized'), (4, 'chunks')]
```

That’s because it uses the default `idx_updater` function just increments by one.
This function, DynamicIndexer.count_increments, is shown below

```pycon
>>> def count_increments(current_idx, data_item, step=1):
...     return current_idx + step
```

To get the index starting at 10, we can specify `start=10`, and to step the
index by 3 we can partialize `count_increments`:

```pycon
>>> from functools import partial
>>> step3 = partial(count_increments, step=3)
>>> list(map(DynamicIndexer(start=10, idx_updater=step3), stream))
[(10, 'stream'), (13, 'of'), (16, 'different'), (19, 'sized'), (22, 'chunks')]
```

You can specify any custom `idx_updater` you want: The requirements being that
this function should take `(current_idx, data_item)` as the input, and
return the next “current index”, that is, what the index of the next data item will
be.
Note that `count_increments` ignored the `data_item` completely, but sometimes
you want to take the data item into account.
For example, your data item may contain several elements, and you want your
index to index these elements, therefore you should update your index by
incrementing it with the number of elements.

We have `DynamicIndexer.size_increments` for that, the code is shown below:

```pycon
>>> def size_increments(current_idx, data_item, size_func=len):
...     return current_idx + size_func(data_item)
>>> size_index = DynamicIndexer(idx_updater=DynamicIndexer.size_increments)
>>> list(map(size_index, stream))
[(0, 'stream'), (6, 'of'), (8, 'different'), (17, 'sized'), (22, 'chunks')]
```

Q: What if I want the index of a data item to be a function of the data item itself?

A: Then you would use that function to make the `(idxof(data_item), data_item)`
pairs directly. `DynamicIndexer` is for the use case where the index of an item
depends on the (number of, sizes of, etc.) items that came before it.

#### *static* count_increments(obj, step=1)

Index updater that ignores `obj` and advances the index by `step`.

```pycon
>>> count_increments(10, 'whatever')
11
>>> count_increments(10, 'whatever', step=5)
15
```

#### idx_updater(obj, step=1)

Index updater that ignores `obj` and advances the index by `step`.

```pycon
>>> count_increments(10, 'whatever')
11
>>> count_increments(10, 'whatever', step=5)
15
```

#### *static* size_increments(obj, size_func=<built-in function len>)

Index updater that advances the index by `size_func(obj)`, `len` by default.

```pycon
>>> size_increments(10, 'abc')
13
```

### *class* creek.tools.Segmenter(buffer, stats_buffer_callback=<function return_buffer_on_stats_condition>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Feed values to a `BufferStats` and hand its contents to a callback after each one.

Each call `seg(new_val)` adds `new_val` to `buffer` (which returns its
statistic), then returns `stats_buffer_callback(stats, list(buffer))`.
With the default callback that is the buffer contents when the statistic is
not `None`, and `None` otherwise.

* **Parameters:**
  * **buffer** ([`BufferStats`](#creek.tools.BufferStats)) – The `BufferStats` instance that stores values and computes the
    statistic.
  * **stats_buffer_callback** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – `(stats, buffer_contents) -> result`;
    `return_buffer_on_stats_condition` by default.

```pycon
>>> gen = iter(range(200))
>>> bs = BufferStats(maxlen=10, func=sum)
>>> return_if_stats_is_odd = partial(
...     return_buffer_on_stats_condition,
...     cond=lambda x: x%2 == 1, else_val='The sum is not odd!'
... )
>>> seg = Segmenter(buffer=bs, stats_buffer_callback=return_if_stats_is_odd)
```

Since the sum of the values in the buffer [1] is odd, the buffer is returned:

```pycon
>>> seg(new_val=1)
[1]
```

Adding `1 + 2` is still odd so:

```pycon
>>> seg(new_val=2)
[1, 2]
```

Now since `1 + 2 + 5` is even, the `else_val` of `return_if_stats_is_odd`
is returned instead

```pycon
>>> seg(new_val=5)
'The sum is not odd!'
```

#### stats_buffer_callback(buffer, cond=<function is_not_none>, else_val=None)

Return `buffer` if `cond(stats)` holds, else `else_val`.

The default `stats_buffer_callback` of `Segmenter`; partialize `cond` and
`else_val` to make your own.

```pycon
>>> return_buffer_on_stats_condition(
... stats=3, buffer=[1,2,3,4], cond=lambda x: x%2 == 1
... )
[1, 2, 3, 4]
>>> return_buffer_on_stats_condition(
... stats=3, buffer=[1,2,3,4], cond=lambda x: x%2 == 0, else_val='3 is not even!'
... )
'3 is not even!'
```

### creek.tools.alt_dynamically_index(idx_updater=<function count_increments>, start=0)

Alternative to dynamically_index using itertools and partial

```pycon
>>> def size_increments(current_idx, data_item, size_func=len):
...     return current_idx + size_func(data_item)
...
>>> stream = ['stream', 'of', 'different', 'sized', 'chunks']
>>> indexer = alt_dynamically_index(size_increments)
>>> t = list(indexer(stream))
>>> assert t == [(0, 'stream'), (6, 'of'), (8, 'different'), (17, 'sized'),
...              (22, 'chunks')]
```

### creek.tools.always_true(x)

Return `True`, whatever `x` is.

### creek.tools.apply_and_fanout(seq, func, idx)

Apply function (that returns an Iterable) to an element of a sequence
and fanout (broadcast) the resulting items, to produce an iterable of tuples each
containing
one of these items along with ‘a copy’ of the other tuple elements

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)]

```pycon
>>> list(apply_and_fanout([1, 'abc', 3], iter, 1))
[(1, 'a', 3), (1, 'b', 3), (1, 'c', 3)]
>>> list(apply_and_fanout(['bob', 'alice', 2], lambda x: x * ['hi'], 2))
[('bob', 'alice', 'hi'), ('bob', 'alice', 'hi')]
>>> list(apply_and_fanout(["bob", "alice", 2], lambda x: x.upper(), 1))
[('bob', 'A', 2), ('bob', 'L', 2), ('bob', 'I', 2), ('bob', 'C', 2), ('bob', 'E', 2)]
```

#### SEE ALSO
`fanout_and_flatten` and `fanout_and_flatten_dicts`

### creek.tools.apply_func_to_index(seq, apply_to_idx, func)

Apply `func` to the element at `apply_to_idx` of `seq`, returning a tuple.

```pycon
>>> apply_func_to_index([1, 2, 3], 1, lambda x: x * 10)
(1, 20, 3)
```

If you’re going to apply the same function to the same index, you might
want to partialize `apply_func_to_index` to be able to reuse it simply:

```pycon
>>> from functools import partial
>>> f = partial(apply_func_to_index, apply_to_idx=0, func=str.upper)
>>> list(map(f, ['abc', 'defgh']))
[('A', 'b', 'c'), ('D', 'e', 'f', 'g', 'h')]
```

### creek.tools.count_increments(current_idx, obj, step=1)

Index updater that ignores `obj` and advances the index by `step`.

```pycon
>>> count_increments(10, 'whatever')
11
>>> count_increments(10, 'whatever', step=5)
15
```

### creek.tools.current_time(current_idx, obj)

Doesn’t even look at current_idx or obj. Just gives the current time

### creek.tools.dynamically_index(iterable, start=0, idx_updater=<function count_increments>)

Generalization of `enumerate(iterable)` that allows one to specify how the
indices should be updated.

The default is the same behavior as `enumerate`: Starts with 0 and increments by 1.

* **Parameters:**
  * **iterable** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)) – The items to index.
  * **start** – The index of the first item.
  * **idx_updater** – `(current_idx, item) -> next_idx`; `count_increments`
    (add 1) by default, `size_increments` adds `len(item)`.
* **Returns:**
  An iterator of `(index, item)` pairs.

#### SEE ALSO
`DynamicIndexer`, the callable doing the work, when you want to apply the
same indexing rule to several streams or keep the current index around.

```pycon
>>> stream = ['stream', 'of', 'different', 'sized', 'chunks']
>>> assert (list(dynamically_index(stream, start=2))
...     == list(enumerate(stream, start=2))
...     == [(2, 'stream'), (3, 'of'), (4, 'different'), (5, 'sized'), (6, 'chunks')]
... )
```

Say we wanted to increment the indices according to the size of the last item
instead of just incrementing by 1 at every iteration tick…

```pycon
>>> def size_increments(current_idx, data_item, size_func=len):
...     return current_idx + size_func(data_item)
>>> size_index = DynamicIndexer(idx_updater=DynamicIndexer.size_increments)
>>> list(map(size_index, stream))
[(0, 'stream'), (6, 'of'), (8, 'different'), (17, 'sized'), (22, 'chunks')]
```

### creek.tools.fanout_and_flatten(iterable_of_seqs, func, idx, aggregator=<built-in method from_iterable of type object>)

Apply apply_and_fanout to an iterable of sequences.

```pycon
>>> seq_iterable = [('abcdef', 'first'), ('ghij', 'second')]
>>> func = lambda a: zip(*([iter(a)] * 2))  # func is a chunker
>>> assert list(fanout_and_flatten(seq_iterable, func, 0)) == [
...     (('a', 'b'), 'first'),
...     (('c', 'd'), 'first'),
...     (('e', 'f'), 'first'),
...     (('g', 'h'), 'second'),
...     (('i', 'j'), 'second')
... ]
```

### creek.tools.fanout_and_flatten_dicts(iterable_of_dicts, func, fields, idx_field, aggregator=<built-in method from_iterable of type object>)

Apply apply_and_fanout to an iterable of dicts.

```pycon
>>> iterable_of_dicts = [
...     {'wf': 'abcdef', 'tag': 'first'}, {'wf': 'ghij', 'tag': 'second'}
... ]
>>> func = lambda a: zip(*([iter(a)] * 2))  # func is a chunker
>>> fields = ['wf', 'tag']
>>> idx_field = 'wf'
>>> assert list(
...     fanout_and_flatten_dicts(iterable_of_dicts, func, fields, idx_field)) == [
...         {'wf': ('a', 'b'), 'tag': 'first'},
...         {'wf': ('c', 'd'), 'tag': 'first'},
...         {'wf': ('e', 'f'), 'tag': 'first'},
...         {'wf': ('g', 'h'), 'tag': 'second'},
...         {'wf': ('i', 'j'), 'tag': 'second'}
... ]
```

### creek.tools.filter_and_index_stream(stream, data_item_filt, timestamper=<class 'enumerate'>)

Index a stream and filter it (based only on the data items).

* **Parameters:**
  * **stream** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)) – The items to index and filter.
  * **data_item_filt** – 

    A predicate on the data item, or a non-callable value:
    ```default
    items equal to that value are then dropped.
    ```
  * **timestamper** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`DataItem`)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`DataItem`)]]) – Turns the stream into `(index, item)` pairs; `enumerate`
    by default.
* **Returns:**
  An iterator of `(index, item)` pairs for the items that pass the filter.

```pycon
>>> assert (
... list(filter_and_index_stream('this  is   a   stream', data_item_filt=' ')) == [
... (0, 't'),
... (1, 'h'),
... (2, 'i'),
... (3, 's'),
... (6, 'i'),
... (7, 's'),
... (11, 'a'),
... (15, 's'),
... (16, 't'),
... (17, 'r'),
... (18, 'e'),
... (19, 'a'),
... (20, 'm')
... ])
```

```pycon
>>> list(filter_and_index_stream(
...     [1, 2, 3, 4, 5, 6, 7, 8],
...     data_item_filt=lambda x: x % 2))
[(0, 1), (2, 3), (4, 5), (6, 7)]
```

### creek.tools.is_not_none(x)

Return `True` if `x` is not `None`.

### creek.tools.return_buffer_on_stats_condition(stats, buffer, cond=<function is_not_none>, else_val=None)

Return `buffer` if `cond(stats)` holds, else `else_val`.

The default `stats_buffer_callback` of `Segmenter`; partialize `cond` and
`else_val` to make your own.

```pycon
>>> return_buffer_on_stats_condition(
... stats=3, buffer=[1,2,3,4], cond=lambda x: x%2 == 1
... )
[1, 2, 3, 4]
>>> return_buffer_on_stats_condition(
... stats=3, buffer=[1,2,3,4], cond=lambda x: x%2 == 0, else_val='3 is not even!'
... )
'3 is not even!'
```

### creek.tools.segment_overlaps(bt_tt_segment, query_bt, query_tt)

Returns True if, and only if, bt_tt_segment overlaps query interval.

A `bt_tt_segment` will need to be of the `(bt, tt, *data)` format.
That is, an iterable of at least two elements (the `bt` and `tt`) followed with
more elements (the actual segment data).

This function is made to be curried, as shown in the following example:

```pycon
>>> from functools import partial
>>> overlapping_segments_filt = partial(segment_overlaps, query_bt=4, query_tt=8)
>>>
>>> list(filter(overlapping_segments_filt, [
...     (1, 3, 'completely before'),
...     (2, 4, 'still completely before (upper bounds are strict)'),
...     (3, 6, 'partially before, but overlaps bottom'),
...     (4, 5, 'totally', 'inside'),  # <- note this tuple has 4 elements
...     (5, 8),  # <- note this tuple has only the minimum (2) elements,
...     (7, 10, 'partially after, but overlaps top'),
...     (8, 11, 'completely after (strict upper bound)'),
...     (100, 101, 'completely after (obviously)')
... ]))
[(3, 6, 'partially before, but overlaps bottom'),
(4, 5, 'totally', 'inside'),
(5, 8),
(7, 10, 'partially after, but overlaps top')]
```

### creek.tools.size_increments(current_idx, obj, size_func=<built-in function len>)

Index updater that advances the index by `size_func(obj)`, `len` by default.

```pycon
>>> size_increments(10, 'abc')
13
```
