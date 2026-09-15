# creek.labeling

Tools to label/annotate stream elements

The motivating example is the case of an incoming stream
that we need to segment, according to the detection of an event.

For example, take a stream of integers and detect the event “multiple of 5”:

```python
1->2->3->4->'multiple of 5'->6->7->...
```

When the stream is “live”, we don’t want to process it immediately, but instead
we prefer to annotate it on the fly, by adding some metadata to it.

The simplest addition of metadata information could look like:

```python
3->4->('multiple of 5', 5) -> 6 -> ...
```

One critique of using  tuples to contain both annotated (here, `5`) and annotation (here, `'multiple of 5'`)
is that the semantics aren’t explicit.
The fact that the original element was annotated the distinction of annotation and annotated is based on an
implicit convention.
This is not too much of a problem here, but becomes unwieldy in more complex situations, for example,
if we want to accommodate for multiple labels.

A `LabeledElement` `x` has an attribute `x.element`,
and a container of labels `x.labels` (list, set or dict).

Multiple labels can be used to segment streams into overlapping segments.

```python
(group0)->(group0)->(group0, group1)->(group0, group1)-> (group1)->(group1)->...
```

### Functions

| [`label_element`](#creek.labeling.label_element)(elem, label, labeled_element_cls)   | Label `element` with `label` (or add this label to the existing labels).   |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|

### Classes

| [`DictLabeledElement`](#creek.labeling.DictLabeledElement)(element)   | A LabeledElement that uses a `dict` as the labels container.                        |
|--------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`LabeledElement`](#creek.labeling.LabeledElement)(element)       | Abstract class to label elements -- that is, associate some metadata to an element. |
| [`ListLabeledElement`](#creek.labeling.ListLabeledElement)(element)   | A LabeledElement that uses a `list` as the labels container.                        |
| [`SetLabeledElement`](#creek.labeling.SetLabeledElement)(element)    | A LabeledElement that uses a `set` as the labels container.                         |

### *class* creek.labeling.DictLabeledElement(element)

Bases: [`LabeledElement`](#creek.labeling.LabeledElement)

A LabeledElement that uses a `dict` as the labels container.
Use this when you need to keep labels classified and have quick access to the
a specific class of labels.
Note that when adding a label, you need to specify it as a `{key: val, ...}`
`dict`, the keys being the (hashable) label kinds,
and the vals being the values for those kinds.

```pycon
>>> x = DictLabeledElement(42).add_label({'string': 'forty-two'})
>>> x.element
42
>>> x.labels
{'string': 'forty-two'}
>>> x.add_label({'type': 'number', 'prime': False})
DictLabeledElement(42)
>>> x.element
42
>>> assert x.labels == {'string': 'forty-two', 'type': 'number', 'prime': False}
```

#### *static* add_new_label(labels, label)

Merge the `label` dict into `labels`.

#### mk_new_labels_container

alias of [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* creek.labeling.LabeledElement(element)

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

Abstract class to label elements – that is, associate some metadata to an element.

To make a concrete LabeledElement, one must subclass LabeledElement and provide

- a `mk_new_labels_container`, a `LabelFactory`, which is a callable that takes
  no input and returns a new empty labels container
- an `add_new_label`, an `AddLabel`, a `(Labels, Label)` callable that adds
  a single label to the labels container.

#### add_label(label)

Add `label` to this element’s labels and return `self`.

#### *abstractmethod static* add_new_label(labels, label)

Add `label` to `labels` in place.

#### *abstractmethod static* mk_new_labels_container(self)

Return a new, empty labels container.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`Label` ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any))]

### *class* creek.labeling.ListLabeledElement(element)

Bases: [`LabeledElement`](#creek.labeling.LabeledElement)

A LabeledElement that uses a `list` as the labels container.
Use this when you need to use unhashable labels, or label insertion order matters,
or don’t need fast `label in labels` checks or label deduplication.

```pycon
>>> x = ListLabeledElement(42).add_label('forty-two')
>>> x.element
42
>>> x.labels
['forty-two']
>>> x.add_label('number')
ListLabeledElement(42)
>>> x.element
42
>>> assert x.labels == ['forty-two', 'number']
```

#### *static* add_new_label(self, object,)

Append object to the end of the list.

#### mk_new_labels_container

alias of [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

### *class* creek.labeling.SetLabeledElement(element)

Bases: [`LabeledElement`](#creek.labeling.LabeledElement)

A LabeledElement that uses a `set` as the labels container.
Use this when you want to get fast `label in labels` check and/or maintain the
labels unduplicated.
Note that since `set` is the container, the labels will have to be hashable.

```pycon
>>> x = SetLabeledElement(42).add_label('forty-two')
>>> x.element
42
>>> x.labels
{'forty-two'}
>>> x.add_label('number')
SetLabeledElement(42)
>>> x.element
42
>>> assert x.labels == {'forty-two', 'number'}
```

#### *static* add_new_label()

Add an element to a set.

This has no effect if the element is already present.

#### mk_new_labels_container

alias of [`set`](https://docs.python.org/3/builtins/stdtypes.html#set)

### creek.labeling.label_element(elem, label, labeled_element_cls)

Label `element` with `label` (or add this label to the existing labels).

The `labeled_element_cls`, the `LabeledElement` class to use to label the element,
is meant to be “partialized out”, like this:

```pycon
>>> from functools import partial
>>> from creek.labeling import DictLabeledElement
>>> my_label_element = partial(label_element, labeled_element_cls=DictLabeledElement)
>>> # and then just use my_label_element(elem, label) to label elem
```

You’ll probably often want to use `DictLabeledElement`, because, for example:

```text
{'n_channels': 2, 'phase', 2, 'session': 16987485}
```

is a lot easier (and less dangerous) to use then, say:

```text
[2, 2, 16987485]
```

But there are cases where, say:

```pycon
>>> from creek.labeling import SetLabeledElement
>>> my_label_element = partial(label_element, labeled_element_cls=SetLabeledElement)
>>> x = my_label_element(42, 'divisible_by_seven')
>>> _ = my_label_element(x, 'is_a_number')
>>> 'divisible_by_seven' in x  # equivalent to 'divisible_by_seven' in x.labels
True
>>> x.labels.issuperset({'is_a_number', 'divisible_by_seven'})
True
```

is more convenient to use then using a dict with boolean values to do the same

* **Parameters:**
  * **elem** (`Union`[`Element` ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)), [`LabeledElement`](#creek.labeling.LabeledElement)]) – The element that is being labeled
  * **label** (`Label` ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any))) – The label to add to the element
  * **labeled_element_cls** – The `LabeledElement` class to use to label the element
* **Return type:**
  [`LabeledElement`](#creek.labeling.LabeledElement)
* **Returns:**
  The labeled element: a new `labeled_element_cls` wrapping `elem`, or
  `elem` itself (with the label added) if it already is a `labeled_element_cls`.
