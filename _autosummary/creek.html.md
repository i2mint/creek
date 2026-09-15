# creek

creek: a simple facade for streams.

Wrap a stream in a `Creek` and override `pre_iter`, `data_to_obj` or
`post_iter` to layer transformations on it; read an unbounded stream as if it
were a list with `InfiniteSeq`; index, window and segment streams with the
functions and classes of `creek.tools`.

Main entry points:

- `Creek`: the layer-able stream wrapper (`creek.base`)
- `InfiniteSeq, IndexedBuffer`: list-like views of unbounded streams (`creek.infinite_sequence`)
- `BufferStats, Segmenter`: rolling-window statistics and segmentation (`creek.tools`)
- `dynamically_index, filter_and_index_stream`: indexing streams (`creek.tools`)

```pycon
>>> from creek import Creek
>>> class Doubler(Creek):
...     def data_to_obj(self, x):
...         return x * 2
>>> list(Doubler([1, 2, 3]))
[2, 4, 6]
```

### Modules

| [`automatas`](creek.automatas.html.md#module-creek.automatas)                 | Automatas (finite state machines): step a state through a stream of symbols.   |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`base`](creek.base.html.md#module-creek.base)                           | The `Creek` base class: a layer-able wrapper around a stream.                  |
| [`infinite_sequence`](creek.infinite_sequence.html.md#module-creek.infinite_sequence) | List-like read access to an unbounded stream, through a bounded buffer.        |
| [`labeling`](creek.labeling.html.md#module-creek.labeling)                   | Tools to label/annotate stream elements                                        |
| [`multi_streams`](creek.multi_streams.html.md#module-creek.multi_streams)         | Merge several sorted streams into one stream of `(stream_id, item)` pairs.     |
| [`tools`](creek.tools.html.md#module-creek.tools)                         | Tools to index, window and segment streams.                                    |
| [`util`](creek.util.html.md#module-creek.util)                           | Iterator, cursor and function-composition utilities used across creek.         |
