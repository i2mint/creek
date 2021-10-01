import pytest
from creek.labeling import (
    DictLabeledElement,
    SetLabeledElement,
    ListLabeledElement,
)


def test_dummy():
    assert True


def test_dict_labeledelement():
    x = DictLabeledElement(42).add_label({'string': 'forty-two'})
    assert x.element == 42
    assert x.labels == {'string': 'forty-two'}
    x.add_label({'type': 'number', 'prime': False})
    assert x.element == 42
    assert x.labels == {
        'string': 'forty-two',
        'type': 'number',
        'prime': False,
    }


def test_set_labeledelement():
    x = SetLabeledElement(42).add_label('forty-two')
    assert x.element == 42
    assert x.labels == {'forty-two'}
    x.add_label('number')
    assert x.element == 42
    assert x.labels == {'forty-two', 'number'}


def test_list_labeledelement():
    x = ListLabeledElement(42).add_label('forty-two')
    assert x.element == 42
    assert x.labels == ['forty-two']
    x.add_label('number')
    assert x.element == 42
    assert x.labels == ['forty-two', 'number']
