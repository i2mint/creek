"""
Tools to label/annotate stream elements

The motivating example is the case of an incoming stream
that we need to segment, according to the detection of an event.

For example, take a stream of integers and detect the event "multiple of 5":

```
1->2->3->4->'multiple of 5'->6->7->...

```

When the stream is "live", we don't want to process it immediately, but instead
we prefer to annotate it on the fly, by adding some metadata to it.

The simplest addition of metadata information could look like:

```
3->4->('multiple of 5', 5) -> 6 -> ...

```

This module treats the more complicated case of "multilabelling":
a LabelledElement x has an attribute **x.element**,
and a container of labels **x.labels** (list, set or dict).

Multilabels can be used to segments streams into overlapping segments.

```
(group0)->(group0)->(group0, group1)->(group0, group1)-> (group1)->(group1)->...

```

"""

from typing import NewType, Iterable, Callable, Any, TypeVar, Union
from abc import ABC, abstractmethod

KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.
Element = NewType('Element', Any)
Label = NewType('Label', Any)
Labels = Iterable[Label]
LabelFactory = Callable[[], Label]
AddLabel = Callable[[Labels, Label], Any]


class LabeledElement(ABC):
    """
    Abstract class to label elements -- that is, associate some metadata to an element.

    To make a concrete LabeledElement, one must subclass LabeledElement and provide

    - a `mk_new_labels_container`, a `LabelFactory`, which is a callable that takes no \
    input and returns a new empty labels container
    - a `add_new_label`, an `AddLabel`, a (Labels, Label) callable that adds a single \
    label to the labels container.
    """

    def __init__(self, element: Element):
        self.element = element
        self.labels = self.mk_new_labels_container()

    @staticmethod
    @abstractmethod
    def mk_new_labels_container(self) -> Labels:
        raise NotImplemented('Need to implement mk_new_labels_container')

    add_new_label: AddLabel

    @staticmethod
    @abstractmethod
    def add_new_label(labels: Labels, label: Label):
        raise NotImplemented('Need to implement add_new_label')

    def __repr__(self):
        return f'{type(self).__name__}({self.element})'

    def add_label(self, label):
        self.add_new_label(self.labels, label)
        return self

    def __contains__(self, label):
        return label in self.labels


class DictLabeledElement(LabeledElement):
    """A LabeledElement that uses a `dict` as the labels container.
    Use this when you need to keep labels classified and have quick access to the
    a specific class of labels.
    Note that when adding a label, you need to specify it as a `{key: val, ...}`
    `dict`, the keys being the (hashable) label kinds,
    and the vals being the values for those kinds.

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
    """

    mk_new_labels_container = staticmethod(dict)

    @staticmethod
    def add_new_label(labels: dict, label: dict):
        labels.update(label)


class SetLabeledElement(LabeledElement):
    """A LabeledElement that uses a `set` as the labels container.
    Use this when you want to get fast `label in labels` check and/or maintain the
    labels unduplicated.
    Note that since `set` is the container, the labels will have to be hashable.

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
    """

    mk_new_labels_container = staticmethod(set)
    add_new_label = staticmethod(set.add)


class ListLabeledElement(LabeledElement):
    """A LabeledElement that uses a `list` as the labels container.
    Use this when you need to use unhashable labels, or label insertion order matters,
    or don't need fast `label in labels` checks or label deduplication.

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
    """

    mk_new_labels_container = staticmethod(list)
    add_new_label = staticmethod(list.append)


def label_element(
    elem: Union[Element, LabeledElement],
    label: Label,
    labeled_element_cls,  # TODO: LabeledElement annotation makes linter complain!?
) -> LabeledElement:
    """Label `element` with `label` (or add this label to the existing labels).

    The `labeled_element_cls`, the `LabeledElement` class to use to label the element,
    is meant to be "partialized out", like this:

    >>> from functools import partial
    >>> from creek.labeling import DictLabeledElement
    >>> my_label_element = partial(label_element, labeled_element_cls=DictLabeledElement)
    >>> # and then just use my_label_element(elem, label) to label elem

    You'll probably often want to use `DictLabeledElement`, because, for example:

    ```
    {'n_channels': 2, 'phase', 2, 'session': 16987485}
    ```

    is a lot easier (and less dangerous) to use then, say:

    ```
    [2, 2, 16987485]
    ```

    But there are cases where, say:

    >>> from creek.labeling import SetLabeledElement
    >>> my_label_element = partial(label_element, labeled_element_cls=SetLabeledElement)
    >>> x = my_label_element(42, 'divisible_by_seven')
    >>> _ = my_label_element(x, 'is_a_number')
    >>> 'divisible_by_seven' in x  # equivalent to 'divisible_by_seven' in x.labels
    True
    >>> x.labels.issuperset({'is_a_number', 'divisible_by_seven'})
    True

    is more convenient to use then using a dict with boolean values to do the same

    :param elem: The element that is being labeled
    :param label: The label to add to the element
    :param labeled_element_cls: The `LabeledElement` class to use to label the element
    :return:
    """
    if not isinstance(elem, labeled_element_cls):
        return labeled_element_cls(elem).add_label(label)
    else:  # elem is already an labeled_element_cls itself, so
        return elem.add_label(label)
