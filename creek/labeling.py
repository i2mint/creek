"""Tools to label/annotate stream elements"""

from typing import NewType, Iterable, Callable, Any, Tuple, TypeVar
from abc import ABC, abstractmethod

KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.
KV = Tuple[KT, VT]  # a (key, value) pair
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


def label_element(elem, label, labeled_element_cls):
    if not isinstance(elem, labeled_element_cls):
        return labeled_element_cls(elem).add_label(label)
    else:  # elem is already an labeled_element_cls itself, so
        return elem.add_label(label)
