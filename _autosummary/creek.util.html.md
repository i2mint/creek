# creek.util

Iterator, cursor and function-composition utilities used across creek.

Main entry points:

- `iterize`: turn an `In -> Out` function into an `Iterator[In] -> Iterator[Out]` one
- `Pipe`: compose functions into one callable
- `iterate_skipping_errors`: iterate, skipping (and optionally reporting) exceptions
- `to_iterator, cursor_to_iterator, iterator_to_cursor`: convert between iterables, iterators and argument-less “cursor” functions

```pycon
>>> from creek.util import iterize
>>> list(iterize(str.upper)(['a', 'b']))
['A', 'B']
```

### Functions

| [`cls_wrap`](#creek.util.cls_wrap)(cls, obj)                                | Wrap `obj` in `cls`: a subclass wrapping instances at construction if `obj` is a class, else `cls(obj)`.   |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| [`cursor_to_iterator`](#creek.util.cursor_to_iterator)(cursor[, sentinel])            | Get an iterator from a cursor function.                                                                    |
| [`identity_func`](#creek.util.identity_func)(x)                                  | Return `x` unchanged.                                                                                      |
| [`iterable_to_cursor`](#creek.util.iterable_to_cursor)(iterable)                      | Get a cursor function from an iterable.                                                                    |
| [`iterable_to_iterator`](#creek.util.iterable_to_iterator)(iterable[, sentinel])        | Return `iter(iterable)`, or, given a sentinel, an iterator stopping just before that value.                |
| [`iterate_skipping_errors`](#creek.util.iterate_skipping_errors)(g[, error_callback, ...]) | Iterate over a generator, skipping errors and calling an error callback if provided.                       |
| [`iterate_until_exception`](#creek.util.iterate_until_exception)(iterator[, ...])          | Call `next` on `iterator` until one of `interrupt_exceptions` is raised, then print `ending`.              |
| [`iterator_to_cursor`](#creek.util.iterator_to_cursor)(iterator[, default])           | Get a cursor function for the input iterator.                                                              |
| [`iterize`](#creek.util.iterize)(func[, name])                             | From an In->Out function, makes a Iterator[In]->Itertor[Out] function.                                     |
| [`static_identity_method`](#creek.util.static_identity_method)(x)                         | Return `x` unchanged.                                                                                      |
| [`to_iterator`](#creek.util.to_iterator)()                                     | Get an iterator from an iterable or a cursor function                                                      |

### Classes

| [`CursorFunc`](#creek.util.CursorFunc)(\*args, \*\*kwargs)   | An argument-less function returning an iterator's values                     |
|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`IterableType`](#creek.util.IterableType)(\*args, \*\*kwargs) | An iterable type that can actually be used in singledispatch                 |
| [`IteratorType`](#creek.util.IteratorType)(\*args, \*\*kwargs) | An iterator type that can actually be used in singledispatch                 |
| [`Pipe`](#creek.util.Pipe)(\*funcs, \*\*named_funcs)   | Simple function composition.                                                 |
| [`PreIter`](#creek.util.PreIter)()                        | Namespace holding `skip_items`, a `pre_iter` that skips the first `n` items. |
| [`stream_util`](#creek.util.stream_util)()                    | A namespace of small stream helpers.                                         |

### Exceptions

| [`IteratorExit`](#creek.util.IteratorExit)   | Raised when an iterator should quit being iterated on, signaling this event any process that cares to catch the signal.   |
|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|

### *class* creek.util.CursorFunc(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

An argument-less function returning an iterator’s values

### *class* creek.util.IterableType(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

An iterable type that can actually be used in singledispatch

```pycon
>>> assert isinstance([1, 2, 3], IterableType)
>>> assert not isinstance(2, IterableType)
```

### *exception* creek.util.IteratorExit

Bases: [`BaseException`](https://docs.python.org/3/builtins/exceptions.html#BaseException)

Raised when an iterator should quit being iterated on, signaling this event
any process that cares to catch the signal.
We chose to inherit directly from `BaseException` instead of `Exception`
for the same reason that `GeneratorExit` does: Because it’s not technically
an error.

See: [https://docs.python.org/3/library/exceptions.html#GeneratorExit](https://docs.python.org/3/library/exceptions.html#GeneratorExit)

### *class* creek.util.IteratorType(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

An iterator type that can actually be used in singledispatch

```pycon
>>> assert isinstance(iter([1, 2, 3]), IteratorType)
>>> assert not isinstance([1, 2, 3], IteratorType)
```

### *class* creek.util.Pipe(\*funcs, \*\*named_funcs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Simple function composition. That is, gives you a callable that implements input -> f_1 -> … -> f_n -> output.

```pycon
>>> def foo(a, b=2):
...     return a + b
>>> f = Pipe(foo, lambda x: print(f"x: {x}"))
>>> f(3)
x: 5
>>> len(f)
2
```

You can name functions, but this would just be for documentation purposes.
The names are completely ignored.

```pycon
>>> g = Pipe(
...     add_numbers = lambda x, y: x + y,
...     multiply_by_2 = lambda x: x * 2,
...     stringify = str
... )
>>> g(2, 3)
'10'
>>> len(g)
3
```

### Notes

- Pipe instances don’t have a \_\_name_\_ etc. So some expectations of normal functions are not met.
- Pipe instance are pickalable (as long as the functions that compose them are)

You can specify a single functions:

```pycon
>>> Pipe(lambda x: x + 1)(2)
3
```

but

```pycon
>>> Pipe()
Traceback (most recent call last):
  ...
ValueError: You need to specify at least one function!
```

You can specify an instance name and/or doc with the special (reserved) argument
names `__name__` and `__doc__` (which therefore can’t be used as function names):

```pycon
>>> f = Pipe(map, add_it=sum, __name__='map_and_sum', __doc__='Apply func and add')
>>> f(lambda x: x * 10, [1, 2, 3])
60
>>> f.__name__
'map_and_sum'
>>> f.__doc__
'Apply func and add'
```

### *class* creek.util.PreIter

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Namespace holding `skip_items`, a `pre_iter` that skips the first `n` items.

#### skip_items(instance, n)

Skip the first `n` items of `instance`.

### creek.util.cls_wrap(cls, obj)

Wrap `obj` in `cls`: a subclass wrapping instances at construction if `obj` is a class, else `cls(obj)`.

### creek.util.cursor_to_iterator(cursor, sentinel=<creek.util.no_sentinel object>)

Get an iterator from a cursor function.

A cursor function is a callable that you call (without arguments) to get items of
data one by one.

Sometimes, especially in live io contexts, that’s the kind interface you’re given
to consume a stream.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)

```pycon
>>> cursor = iter([1, 2, 3]).__next__
>>> assert not isinstance(cursor, Iterator)
>>> assert not isinstance(cursor, Iterable)
>>> assert callable(cursor)
```

If you want to consume your stream as an iterator instead, use `cursor_to_iterator`.

```pycon
>>> iterator = cursor_to_iterator(cursor)
>>> assert isinstance(iterator, Iterator)
>>> list(iterator)
[1, 2, 3]
```

If you want your iterator to stop (without a fuss) as soon as the cursor returns a
particular element (called a sentinel), say it:

```pycon
>>> cursor = iter([1, 2, None, None, 3]).__next__
>>> iterator = cursor_to_iterator(cursor, sentinel=None)
>>> list(iterator)
[1, 2]
```

### creek.util.identity_func(x)

Return `x` unchanged.

### creek.util.iterable_to_cursor(iterable)

Get a cursor function from an iterable.

* **Return type:**
  [`CursorFunc`](#creek.util.CursorFunc)

### creek.util.iterable_to_iterator(iterable, sentinel=<creek.util.no_sentinel object>)

Return `iter(iterable)`, or, given a sentinel, an iterator stopping just before that value.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)

```pycon
>>> iterable = [1, 2, 3]
>>> iterator = iterable_to_iterator(iterable)
>>> assert isinstance(iterator, Iterator)
>>> assert list(iterator) == iterable
```

You can also specify a sentinel, which will result in the iterator stoping just
before it encounters that sentinel value

```pycon
>>> iterable = [1, 2, 3, 4, None, None, 7]
>>> iterator = iterable_to_iterator(iterable, None)
>>> assert isinstance(iterator, Iterator)
>>> list(iterator)
[1, 2, 3, 4]
```

### creek.util.iterate_skipping_errors(g, error_callback=None, caught_exceptions=(<class 'Exception'>, ))

Iterate over a generator, skipping errors and calling an error callback if provided.

* **Parameters:**
  * **g** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)) – The generator to iterate over
  * **error_callback** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`BaseException`](https://docs.python.org/3/builtins/exceptions.html#BaseException)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A callback to call when an error is encountered.
  * **caught_exceptions** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`BaseException`](https://docs.python.org/3/builtins/exceptions.html#BaseException)]) – The exceptions to catch and skip.
* **Return type:**
  [`Generator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)
* **Returns:**
  A generator that yields the values of the original generator,
  skipping errors.

```pycon
>>> list(iterate_skipping_errors(map(lambda x: 1 / x, [1, 0, 2])))
[1.0, 0.5]
>>> list(iterate_skipping_errors(map(lambda x: 1 / x, [1, 0, 2]), print))
division by zero
[1.0, 0.5]
```

See [https://github.com/i2mint/creek/issues/6](https://github.com/i2mint/creek/issues/6) for more info.

### creek.util.iterate_until_exception(iterator, interrupt_exceptions=(<class 'StopIteration'>, <class 'creek.util.IteratorExit'>, <class 'KeyboardInterrupt'>))

Call `next` on `iterator` until one of `interrupt_exceptions` is raised, then print `ending`.

### creek.util.iterator_to_cursor(iterator, default=<creek.util.no_default object>)

Get a cursor function for the input iterator.

* **Return type:**
  [`CursorFunc`](#creek.util.CursorFunc)

```pycon
>>> iterator = iter([1, 2, 3])
>>> cursor = iterator_to_cursor(iterator)
>>> assert callable(cursor)
>>> assert cursor() == 1
>>> assert list(cursor_to_iterator(cursor)) == [2, 3]
```

Note how we consumed the cursor till the end; by using cursor_to_iterator.
Indeed, `list(iter(cursor))` wouldn’t have worked since a cursor isn’t a iterator,
but a callable to get the items an the iterator would give you.

You can specify a default. The default has the same role that it has for the
`next` function: It makes the cursor function return that default when the iterator
has been “consumed” (i.e. would raise a `StopIteration`).

```pycon
>>> iterator = iter([1, 2])
>>> cursor = iterator_to_cursor(iterator, None)
>>> assert callable(cursor)
>>> cursor()
1
>>> cursor()
2
```

And then…

```pycon
>>> assert cursor() is None
>>> assert cursor() is None
```

forever.

### creek.util.iterize(func, name=None)

From an In->Out function, makes a Iterator[In]->Itertor[Out] function.

```pycon
>>> f = lambda x: x * 10
>>> f(2)
20
>>> iterized_f = iterize(f)
>>> list(iterized_f(iter([1,2,3])))
[10, 20, 30]
```

### creek.util.static_identity_method(x)

Return `x` unchanged.

### *class* creek.util.stream_util

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A namespace of small stream helpers.

#### always_true(\*\*kwargs)

Return `True`.

#### do_nothing(\*\*kwargs)

Return `None`.

#### rewind(instance)

Seek `instance` back to 0.

#### skip_lines(instance, n_lines_to_skip=0)

Seek `instance` back to 0 (`n_lines_to_skip` is currently ignored).

### creek.util.to_iterator(x, sentinel=<creek.util.no_sentinel object>)

### creek.util.to_iterator(x, sentinel=<creek.util.no_sentinel object>)

### creek.util.to_iterator(x, sentinel=<creek.util.no_sentinel object>)

Get an iterator from an iterable or a cursor function

```pycon
>>> from typing import Iterator
>>> it = to_iterator([1, 2, 3])
>>> assert isinstance(it, Iterator)
>>> list(it)
[1, 2, 3]
>>> list(it)
[]
```

```pycon
>>> cursor = iter([1, 2, 3]).__next__
>>> assert isinstance(cursor, CursorFunc)
>>> it = to_iterator(cursor)
>>> assert isinstance(it, Iterator)
>>> list(it)
[1, 2, 3]
>>> list(it)
[]
```

You can use sentinels too

```pycon
>>> list(to_iterator([1, 2, None, 4], sentinel=None))
[1, 2]
>>> cursor = iter([1, 2, 3, 4, 5]).__next__
>>> list(to_iterator(cursor, sentinel=4))
[1, 2, 3]
```
