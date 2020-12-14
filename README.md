# creek
Simple streams facade.

To install:	```pip install creek```

The ``Creek`` base class offsers a layer-able wrap of the stream interface.

There are three layering methods -- pre_iter, data_to_obj, and post_filt
-- whose use is demonstrated in the iteration code below:

```
for line in self.pre_iter(self.stream):  # pre_iter: prepare and/or filter the stream
    obj = self.data_to_obj(line)  # data_to_obj: Transforms the data that stream yields
    if self.post_filt(obj):  # post_filt: Filters the stream further (but based on object now)
        yield obj
```

Examples:



```pydocstring
>>> from io import StringIO
>>> src = StringIO(
... '''a, b, c
... 1,2, 3
... 4, 5,6
... '''
... )
>>>
>>> from creek import Creek
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
```

Let's add a filter! There's two kinds you can use.
One that is applied to the line before the data is transformed by data_to_obj,
and the other that is applied after (to the obj).

```pydocstring
>>> from creek import Creek
>>> from io import StringIO
>>>
>>> src = StringIO(
...     '''a, b, c
... 1,2, 3
... 4, 5,6
... ''')
>>> class MyFilteredCreek(MyCreek):
...     def post_filt(self, obj):
...         return str.isnumeric(obj[0])
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
```

Recipes:
- pre_iter: involving itertools.islice to skip header lines
- pre_iter: involving enumerate to get line indices in stream iterator
- pre_iter = functools.partial(map, line_pre_proc_func) to preprocess all lines with line_pre_proc_func
- pre_iter: include filter before obj