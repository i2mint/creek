"""Coverage-pass tests for ``creek.automatas`` and ``creek.util``."""

import io

import pytest

from creek.automatas import (
    BasicAutomata,
    MappingTransitionFunc,
    automata,
    basic_automata,
    mapping_to_transition_func,
)
from creek.util import PreIter, cls_wrap, stream_util

TRANSITIONS = {("a", 0): "b", ("b", 1): "a", ("a", 1): "a"}


def test_mapping_to_transition_func_strict_and_lenient():
    strict = mapping_to_transition_func(TRANSITIONS)
    assert strict("a", 0) == "b"
    with pytest.raises(KeyError):
        strict("b", 0)
    lenient = mapping_to_transition_func(TRANSITIONS, strict=False)
    assert lenient("b", 0) == "b"  # unmapped: the state is unchanged


@pytest.mark.parametrize("strict", [True, False])
def test_mapping_transition_func_class_matches_function(strict):
    cls_form = MappingTransitionFunc(TRANSITIONS, strict=strict)
    fn_form = mapping_to_transition_func(TRANSITIONS, strict=strict)
    assert cls_form("a", 1) == fn_form("a", 1) == "a"
    if not strict:
        assert cls_form("b", 0) == "b"
    else:
        with pytest.raises(KeyError):
            cls_form("b", 0)


def test_basic_automata_functional_and_alias():
    f = mapping_to_transition_func(TRANSITIONS)
    assert list(basic_automata(f, "a")([0, 1, 1])) == ["b", "a", "a"]
    assert automata is basic_automata


def test_basic_automata_class_state_and_reset():
    a = BasicAutomata(mapping_to_transition_func(TRANSITIONS), state="a")
    assert list(a("a", [0, 1])) == ["b", "a"]
    assert a.state == "a"
    assert a.transition(0) == "b" and a.state == "b"
    assert a.reset() is a and a.state == "a"


@pytest.mark.xfail(
    reason="map_states: the comprehension rebinds ``state`` to the next state, so "
    "keys are built from the next state (\{('A', 0): 'B'\} is expected)",
    strict=True,
)
def test_mapping_transition_func_map_states_maps_next_states():
    mapped = MappingTransitionFunc({("a", 0): "b"}).map_states(str.upper)
    assert mapped("A", 0) == "B"


def test_mapping_transition_func_map_states_with_a_mapping_runs():
    mapped = MappingTransitionFunc({("a", 0): "b"}).map_states({"a": "x", "b": "y"})
    assert isinstance(mapped, MappingTransitionFunc) and mapped.strict
    assert len(mapped.mapping) == 1


def test_stream_util_helpers():
    assert stream_util.always_true(1, k=2) is True
    assert stream_util.do_nothing(1, k=2) is None
    f = io.StringIO("a\nb\n")
    f.read()
    stream_util().rewind(f)
    assert f.read() == "a\nb\n"
    f.read()
    stream_util().skip_lines(f, 3)
    assert f.tell() == 0


def test_pre_iter_skip_items():
    assert list(PreIter().skip_items(iter(range(5)), 2)) == [2, 3, 4]


def test_cls_wrap_instance_and_class():
    class Box:
        def __init__(self, x):
            self.x = x

    class Plain:
        def __init__(self, a, b=2):
            self.total = a + b

    wrapped = cls_wrap(Box, 5)
    assert isinstance(wrapped, Box) and wrapped.x == 5

    Wrapped = cls_wrap(Box, Plain)
    w = Wrapped(1, b=3)
    assert isinstance(w, Box) and w.x.total == 4
    assert Wrapped.__name__ == "Plain"
