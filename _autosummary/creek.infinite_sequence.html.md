# creek.infinite_sequence

List-like read access to an unbounded stream, through a bounded buffer.

An `IndexedBuffer` keeps the last `buffer_len` items of a stream and lets you
index them with the positions they had in the stream. An `InfiniteSeq` pairs such
a buffer with an iterator, pulling items on demand, so that `s[i:j]` behaves as
if the whole stream were a list, as long as queries move forward and fit in the
buffer.

Main entry points:

- `InfiniteSeq`: an iterator plus a buffer, sliced like a list
- `IndexedBuffer`: the buffer alone, fed with `append` / `extend`
- `BufferedGetter`: a buffer queried with a filter function instead of indices

```pycon
>>> from itertools import count
>>> from creek.infinite_sequence import InfiniteSeq
>>> s = InfiniteSeq(count(), buffer_len=5)
>>> s[3:6]
[3, 4, 5]
```

### Functions

| [`absolute_item`](#creek.infinite_sequence.absolute_item)(item, max_idx)              | Returns an item with absolute references: i.e. with negative indices idx resolved to max_idx + idx.   |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| [`asis`](#creek.infinite_sequence.asis)(obj)                                 | Return `obj` unchanged (the default transform).                                                       |
| [`consume`](#creek.infinite_sequence.consume)(gen, n)                           | Consume n iterations of generator (without returning elements)                                        |
| [`new_type`](#creek.infinite_sequence.new_type)(name, typ[, doc])                | Make a `typing.NewType` called `name` with `doc` as its docstring (`typ` is currently ignored).       |
| [`none_safe_addition`](#creek.infinite_sequence.none_safe_addition)(x, y)                  | Adds the two numbers if x is not None, or return None if not                                          |
| [`shift_slice`](#creek.infinite_sequence.shift_slice)(slice_obj, shift)             | Return `slice_obj` with `start` and `stop` shifted by `shift` (`None` bounds stay `None`).            |
| [`simple_interval_relationship`](#creek.infinite_sequence.simple_interval_relationship)(x, y[, ...]) | Get the simple relationship between intervals x and y.                                                |
| [`slice_args`](#creek.infinite_sequence.slice_args)(slice_obj)                     | Return the `(start, stop, step)` of a slice.                                                          |
| [`validate_interval`](#creek.infinite_sequence.validate_interval)(interval)               | Asserts that input is a valid interval, raising a ValueError if not                                   |

### Classes

| [`BufferedGetter`](#creek.infinite_sequence.BufferedGetter)(buffer_len[, prefill, ...])       | `BufferedGetter` is intended to be a more general (but not optimized) class that offers a query-interface to a buffer, intended to be used when the buffer is being filled by a (possibly live) stream of data items.   |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ExceptionRaiserCallbackMixin`](#creek.infinite_sequence.ExceptionRaiserCallbackMixin)(\*args, \*\*kwargs) | Make the instance callable and have the effect of raising the instance.                                                                                                                                                 |
| [`IndexedBuffer`](#creek.infinite_sequence.IndexedBuffer)(buffer_len[, prefill, ...])        | A list-like object that gives a limited-past read view of an unbounded stream.                                                                                                                                          |
| [`InfiniteSeq`](#creek.infinite_sequence.InfiniteSeq)(iterator, buffer_len)                | A list-like (read) view of an unbounded sequence/stream.                                                                                                                                                                |
| [`Relations`](#creek.infinite_sequence.Relations)(\*values)                              | Point-interval and interval-interval relations.                                                                                                                                                                         |

### Exceptions

| [`NotDuringError`](#creek.infinite_sequence.NotDuringError)(\*args, \*\*kwargs)      | IndexError that indicates that there was an attempt to index some data that is not contained in the buffer (i.e. is that a part of the request is NO LONGER, or NOT YET covered by the buffer).   |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`OverlapsFutureError`](#creek.infinite_sequence.OverlapsFutureError)(\*args, \*\*kwargs) | IndexError that indicates that there was an attempt to index some data that is in the FUTURE (i.e. is NOT YET completely covered by the buffer).                                                  |
| [`OverlapsPastError`](#creek.infinite_sequence.OverlapsPastError)(\*args, \*\*kwargs)   | IndexError that indicates that there was an attempt to index some data that is in the PAST (i.e. is NO LONGER completely covered by the buffer).                                                  |
| [`RelationNotHandledError`](#creek.infinite_sequence.RelationNotHandledError)                 | TypeError that indicates that a relation is either not a valid one, or not handled by conditional clause.                                                                                         |

### *class* creek.infinite_sequence.BufferedGetter(buffer_len, prefill=(), input_data_trans=<function asis>, query_trans=<function asis>, slice_get_postproc=<class 'list'>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

`BufferedGetter` is intended to be a more general (but not optimized) class that
offers a query-interface to a buffer, intended to be used when the buffer is
being filled by a (possibly live) stream of data items.

By contrast…
The `IndexedBuffer` is a particular case where the queries are slices and the index
that is sliced on is an enumeration one.
The `InfiniteSeq` is a class combining `IndexedBuffer` with a data source it can
pull data from (according to the demands of the query).

```pycon
>>> from creek.infinite_sequence import BufferedGetter
>>>
>>>
>>> b = BufferedGetter(20)
>>> b.extend([
...     (1, 3, 'completely before'),
...     (2, 4, 'still completely before (upper bounds are strict)'),
...     (3, 6, 'partially before, but overlaps bottom'),
...     (4, 5, 'totally', 'inside'),  # <- note this tuple has 4 elements
...     (5, 8),  # <- note this tuple has only the minimum (2) elements,
...     (7, 10, 'partially after, but overlaps top'),
...     (8, 11, 'completely after (strict upper bound)'),
...     (100, 101, 'completely after (obviously)')
... ])
>>> b[lambda x: 3 < x[0] < 8]
[(4, 5, 'totally', 'inside'),
 (5, 8),
 (7, 10, 'partially after, but overlaps top')]
```

#### append(x)

Transform `x` with `input_data_trans` and add it to the buffer.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### clear()

Empty the buffer.

#### extend(iterable)

Append each item of `iterable`.

#### filter(filt)

Iterate over the buffer items for which `filt` is true.

#### ingress(x)

Return `x` unchanged.

* **Return type:**
  `BufferItem` ([`type`](https://docs.python.org/3/builtins/functions.html#type))

### *class* creek.infinite_sequence.ExceptionRaiserCallbackMixin(\*args, \*\*kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Make the instance callable and have the effect of raising the instance.
Meant to add to an exception class so that instances of this class can be used as callbacks that raise the error

### *class* creek.infinite_sequence.IndexedBuffer(buffer_len, prefill=(), if_overlaps_past=OverlapsPastError('Some of the data requested was in the past or in the future'), if_overlaps_future=OverlapsFutureError('Some of the data requested was in the past or in the future'), slice_get_postproc=<class 'list'>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A list-like object that gives a limited-past read view of an unbounded stream.

For example, say we had the stream of increasing integers 0, 1, 2, …
that is being fed to an `IndexedBuffer`.

What `IndexedBuffer(buffer_len=4)` offers is access to the buffer’s contents,
but using the indices that the stream (if it were one big list in memory) would
use, instead of the buffer’s own indices:

```default
0 1 2 3 [4 5 6 7] 8 9
```

`IndexedBuffer` uses `collections.deque`, exposing the `append`, `extend`
and `clear` methods, updating the index reference under a lock.

* **Parameters:**
  * **buffer_len** – How many of the most recent items are kept.
  * **prefill** – Items the buffer starts with (`max_idx` still starts at 0).
  * **if_overlaps_past** – Stored as an attribute; not consulted by item access,
    which always raises `OverlapsPastError`.
  * **if_overlaps_future** – Stored as an attribute; not consulted by item access,
    which always raises `OverlapsFutureError`.
  * **slice_get_postproc** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – Applied to the `islice` of the buffer that a slice
    request selects; `list` by default.
* **Raises:**
  * [**OverlapsPastError**](#creek.infinite_sequence.OverlapsPastError) – On `s[item]` when some of the requested range is no
        longer in the buffer.
  * [**OverlapsFutureError**](#creek.infinite_sequence.OverlapsFutureError) – On `s[item]` when some of the requested range is not
        yet in the buffer.

```pycon
>>> s = IndexedBuffer(buffer_len=4)
>>> s.extend(range(4))  # adding 4 elements in bulk (filling the buffer completely)
>>> list(s)
[0, 1, 2, 3]
>>> s[2]
2
>>> s[1:2]
[1]
>>> s[1:1]
[]
```

Let’s add two more elements (using append this time), making the buffer “shift”

```pycon
>>> s.append(4)
>>> s.append(5)
>>> list(s)
[2, 3, 4, 5]
>>> s[2]
2
>>> s[5]
5
>>> s[2:5]
[2, 3, 4]
>>> s[3:6]
[3, 4, 5]
>>> assert s[2:6] == list(range(2, 6))
```

You can slice with step:

```pycon
>>> s[2:6:2]
[2, 4]
```

You can slice with negatives

```pycon
>>> s[2:-2]
[2, 3]
```

On the other hand, if you ask for something that is not in the buffer (anymore, or yet), you’ll get an
error that tells you so:

```pycon
>>> # element for idx 1 is missing in [2, 3, 4, 5]
>>> s[1:4]
Traceback (most recent call last):
    ...
OverlapsPastError: You asked for slice(1, 4, None), but the buffer only contains the index range: 2:6
```

```pycon
>>> # elements for 0:2 are missing (as well as 6:9, but OverlapsPastError trumps OverlapsFutureError
>>> s[0:9]
Traceback (most recent call last):
    ...
OverlapsPastError: You asked for slice(0, 9, None), but the buffer only contains the index range: 2:6
```

```pycon
>>> # element for 6:9 are missing in [2, 3, 4, 5]
>>> s[4:9]
Traceback (most recent call last):
    ...
OverlapsFutureError: You asked for slice(4, 9, None), but the buffer only contains the index range: 2:6
```

#### append(x)

Add one item, advancing `max_idx` by 1.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### clear()

Empty the buffer and reset `max_idx` to 0.

#### extend(iterable)

Extend buffer with an iterable of items

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *property* min_idx

The stream index of the oldest item still in the buffer.

#### outer_to_buffer_idx(idx)

Translate a stream index (int, slice, or iterable of ints) into the buffer’s own index.

### *class* creek.infinite_sequence.InfiniteSeq(iterator, buffer_len)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A list-like (read) view of an unbounded sequence/stream.

It is the combination of `IndexedBuffer` and an iterator that will be used to
source the buffer according to the slices that are requested.

If a slice is requested whose data is “in the future”, the iterator will be
consumed until the buffer can satisfy that request.
If the requested slice has any part of it that is “in the past”, that is,
has already been iterated through and is not in the buffer anymore, a
`OverlapsPastError` will be raised.

Therefore, `InfiniteSeq` is meant for ordered slice queries of size no more than
the buffer size.
If these conditions are satisfied, an `InfiniteSeq` will behave (with `i:j`
queries) as if it were one long list in memory.

Can be used with a live stream of data as long as the buffer size is big enough
to handle the data production and query rates.

* **Parameters:**
  * **iterator** ([`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)) – The source of items; consumed forward only, as far as queries
    require.
  * **buffer_len** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – How many of the most recent items stay available.
* **Raises:**
  [**OverlapsPastError**](#creek.infinite_sequence.OverlapsPastError) – On `s[item]` if some of the requested range has already
      left the buffer.

For example, take an iterator that cycles from 0 to 99 forever:

```pycon
>>> from itertools import cycle
>>> iterator = cycle(range(100))
```

Let’s make an `InfiniteSeq` instance for this stream, accomodating for a view of
up to 11 items.

```pycon
>>> s = InfiniteSeq(iterator, buffer_len=11)
```

Let’s ask for element 15 (which is the (15 + 1)th element (and should have a value of 15).

```pycon
>>> s[15]
15
```

Now, to get this value, the iterator will move forward up to that point;
that is, until the buffer’s head (i.e. most recent) item contains that requested (15 + 1)th element.
But the buffer is of size 11, so we still have access to a few previous elements:

```pycon
>>> s[11]
11
>>> s[5:15]
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
```

But if we asked for anything before index 5…

```pycon
>>> s[2:7]
Traceback (most recent call last):
    ...
OverlapsPastError: You asked for slice(2, 7, None), but the buffer only contains the index range: 5:16
```

So we can’t go backwards. But we can always go forwards:

```pycon
>>> s[95:105]
[95, 96, 97, 98, 99, 0, 1, 2, 3, 4]
```

You can also use slices with step and with negative integers (referencing the head of the buffer)

```pycon
>>> s[120:130:2]
[20, 22, 24, 26, 28]
>>> s[120:130]
[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
>>> s[-8:-2]
[22, 23, 24, 25, 26, 27]
```

but you cannot slice farther back than the buffer

```pycon
>>> try:
...     s[-20:-2]
... except OverlapsPastError as e:
...     msg_text = str(e)
>>> print(msg_text)
You asked for slice(110, 128, None), but the buffer only contains the index range: 119:130
```

Sometimes the source provides data in chunks. Sometimes these chunks are not even of fixed size.
In those situations, you can use `itertools.chain` to “flatten” the iterator as in the following example:

```pycon
>>> from creek.infinite_sequence import InfiniteSeq
>>> from typing import Mapping
>>>
>>> class Source(Mapping):
...     n = 100
...
...     __len__ = lambda self: self.n
...
...     def __iter__(self):
...         yield from range(self.n)
...
...     def __getitem__(self, k):
...         print(f"Asking for {k}")
...         return list(range(k * 10, (k + 1) * 10))
...
>>>
>>> source = Source()
>>>
```

See that when we ask for a chunk of data, there’s a print notification about it.

```pycon
>>> assert source[3] == [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
Asking for 3
```

Now let’s make an iterator of the data and an InfiniteSeq (with buffer length 10) on top of it.

```pycon
>>> from itertools import chain
>>> iterator = chain.from_iterable(source.values())
>>> s = InfiniteSeq(iterator, 10)
```

See that when you ask for :5, you see that chunk 0 is requested.

```pycon
>>> s[:5]
Asking for 0
[0, 1, 2, 3, 4]
```

If you ask for something that’s already in the buffer, you won’t see the print notification though.

```pycon
>>> s[4:8]
[4, 5, 6, 7]
```

The following shows you how InfiniteSeq “hits” the data source as it’s getting the data it needs for the request.

```pycon
>>> s[8:12]
Asking for 1
[8, 9, 10, 11]
>>>
>>> s[40:42]
Asking for 2
Asking for 3
Asking for 4
[40, 41]
```

### *exception* creek.infinite_sequence.NotDuringError(\*args, \*\*kwargs)

Bases: [`ExceptionRaiserCallbackMixin`](#creek.infinite_sequence.ExceptionRaiserCallbackMixin), [`IndexError`](https://docs.python.org/3/builtins/exceptions.html#IndexError)

IndexError that indicates that there was an attempt to index some data that is not contained in the buffer
(i.e. is that a part of the request is NO LONGER, or NOT YET covered by the buffer)

### *exception* creek.infinite_sequence.OverlapsFutureError(\*args, \*\*kwargs)

Bases: [`NotDuringError`](#creek.infinite_sequence.NotDuringError)

IndexError that indicates that there was an attempt to index some data that is in the FUTURE
(i.e. is NOT YET completely covered by the buffer)

### *exception* creek.infinite_sequence.OverlapsPastError(\*args, \*\*kwargs)

Bases: [`NotDuringError`](#creek.infinite_sequence.NotDuringError)

IndexError that indicates that there was an attempt to index some data that is in the PAST
(i.e. is NO LONGER completely covered by the buffer)

### *exception* creek.infinite_sequence.RelationNotHandledError

Bases: [`TypeError`](https://docs.python.org/3/builtins/exceptions.html#TypeError)

TypeError that indicates that a relation is either not a valid one, or not handled by conditional clause.

### *class* creek.infinite_sequence.Relations(\*values)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Point-interval and interval-interval relations.

See Allen’s interval algebra for (some of the) interval relations
([https://en.wikipedia.org/wiki/Allen%27s_interval_algebra](https://en.wikipedia.org/wiki/Allen%27s_interval_algebra)).

### creek.infinite_sequence.absolute_item(item, max_idx)

Returns an item with absolute references: i.e. with negative indices idx
resolved to max_idx + idx

```pycon
>>> absolute_item(-1, 10)
9
>>> absolute_item(slice(-4, -2, 2), 10)
slice(6, 8, 2)
```

But anything else that’s not a slice or int will be left untouched
(and will probably result in errors if you use with IndexedBuffer)

```pycon
>>> absolute_item((-7, -2), 10)
(-7, -2)
```

### creek.infinite_sequence.asis(obj)

Return `obj` unchanged (the default transform).

### creek.infinite_sequence.consume(gen, n)

Consume n iterations of generator (without returning elements)

### creek.infinite_sequence.new_type(name, typ, doc=None)

Make a `typing.NewType` called `name` with `doc` as its docstring (`typ` is currently ignored).

### creek.infinite_sequence.none_safe_addition(x, y)

Adds the two numbers if x is not None, or return None if not

### creek.infinite_sequence.shift_slice(slice_obj, shift)

Return `slice_obj` with `start` and `stop` shifted by `shift` (`None` bounds stay `None`).

### creek.infinite_sequence.simple_interval_relationship(x, y, above_bt=<built-in function ge>, below_tt=<built-in function lt>)

Get the simple relationship between intervals x and y.

* **Parameters:**
  * **x** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`Union`[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float)], `Union`[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – A point (a number), an interval (a 2-tuple of numbers), or a slice.
  * **y** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`Union`[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float)], `Union`[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – An interval; a 2-tuple of numbers.
  * **above_bt** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – `above_bt(x_bt, y_bt)` boolean function (`ge` or `gt`)
    deciding if x starts after y does.
  * **below_tt** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – `below_tt(x_tt, y_tt)` boolean function (`lt` or `le`)
    deciding if x ends before y does.
* **Returns:**
  One of three relations:
  `Relations.BEFORE` if some of x is below y,
  `Relations.AFTER` if some of x is after y,
  `Relations.DURING` if x is entirely within y
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `y` (or an interval `x`) is not a 2-tuple with `bt <= tt`.

The target `y` interval is expressed only by its bounds, but we don’t know if
these are inclusive or not. The `above_bt` and `below_tt` arguments let us
express that, by defining what “below the lowest (bt) bound” and “above the
highest (tt) bound” mean.

The function is meant to be curried (partial), for example:

```pycon
>>> from functools import partial
>>> from operator import le, lt, ge, gt
>>> default = simple_interval_relationship  # uses above_bt=ge, below_tt=lt
>>> including_bounds = partial(simple_interval_relationship, above_bt=ge, below_tt=le)
>>> excluding_bounds = partial(simple_interval_relationship, above_bt=gt, below_tt=lt)
```

Take `(4, 8)` as the target interval, and want to query the relationship of other
points and intervals with it.
No matter what the function is, they will always agree on any intervals that don’t
share any bounds.

```pycon
>>> for relation_func in (default, including_bounds, excluding_bounds):
...     print (
...         relation_func(3, (4, 8)),
...         relation_func(5, (4, 8)),
...         relation_func(9, (4, 8)),
...         relation_func((3, 7), (4, 8)),
...         relation_func((5, 7), (4, 8)),
...         relation_func((7, 9), (4, 8))
... )
Relations.BEFORE Relations.DURING Relations.AFTER Relations.BEFORE Relations.DURING Relations.AFTER
Relations.BEFORE Relations.DURING Relations.AFTER Relations.BEFORE Relations.DURING Relations.AFTER
Relations.BEFORE Relations.DURING Relations.AFTER Relations.BEFORE Relations.DURING Relations.AFTER
```

But if the two intervals share some bounds, these functions will diverge.

```pycon
>>> for relation_func in (default, including_bounds, excluding_bounds):
...     print (
...         relation_func(4, (4, 8)),
...         relation_func(8, (4, 8)),
...         relation_func((4, 7), (4, 8)),
...         relation_func((4, 8), (4, 8)),
...         relation_func((5, 8), (4, 8))
... )
Relations.DURING Relations.AFTER Relations.DURING Relations.AFTER Relations.AFTER
Relations.DURING Relations.DURING Relations.DURING Relations.DURING Relations.DURING
Relations.BEFORE Relations.AFTER Relations.BEFORE Relations.BEFORE Relations.AFTER
```

The function can be used with the FIRST argument being a slice object as well.
This can then be used to enable [i:j] access.

```pycon
>>> for relation_func in (default, including_bounds, excluding_bounds):
...     print (
...         relation_func(slice(4, 7), (4, 8)),
...         relation_func(slice(4, 8), (4, 8)),
...         relation_func(slice(5, 8), (4, 8))
... )
Relations.DURING Relations.AFTER Relations.AFTER
Relations.DURING Relations.DURING Relations.DURING
Relations.BEFORE Relations.BEFORE Relations.AFTER
```

### creek.infinite_sequence.slice_args(slice_obj)

Return the `(start, stop, step)` of a slice.

### creek.infinite_sequence.validate_interval(interval)

Asserts that input is a valid interval, raising a ValueError if not
