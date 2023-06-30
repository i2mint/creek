"""Test automatas."""
import pytest
from creek.automatas import (
    mapping_to_transition_func,
    State,
    Symbol,
    basic_automata,
    BasicAutomata,
)


@pytest.mark.parametrize("automata", [basic_automata, BasicAutomata])
def test_automata(automata):
    # A automata that even number of 0s and 1s
    transition_func = mapping_to_transition_func(
        {
            ("even", 0): "odd",
            ("odd", 1): "even",
        },
        strict=False,  # so that all (state, symbol) combinations not listed have no effect
    )
    fa = automata(transition_func)
    symbols = [0, 1, 0, 1, 1]
    it = fa("even", symbols)
    next(it) == "even"  # reads 0, stays on state 'even'
    next(it) == "odd"  # reads 1, applies the ('even', 0): 'odd' transition rule
    next(it) == "odd"  # reads 0, stays on state 'odd'
    next(it) == "even"  # reads 1, applies the ('odd', 1): 'even' transition rule
    next(it) == "odd"  # reads 1, applies the ('even', 0): 'odd' transition rule

    # We can feed the automata with any iterable, and gather the "trace" of all the states
    # it went through like follows. Notice that we can really put any symbol in the
    # iterable, not just 0s and 1s. The automata will just ignore the symbols (remainingin
    # in the same state) if there is no transition for the current state and the symbol.
    assert list(fa("even", [0, 1, 0, 42, "not_even_a_number", 1])) == [
        "odd",
        "even",
        "odd",
        "odd",
        "odd",
        "even",
    ]

    # When the automata only works on a finite set of states and symbols, we call it
    # a "finite automata". Above, we could have specified strict=True and explicitly
    # list all possible combinations in the transition mapping to get a finite automata.
    # Here's an example where neither states nor symbols have to be from a finite set
    def my_special_transition(state: State, symbol: Symbol) -> State:
        if state in symbol or symbol in state:
            # if the symbol is part of, or contains the state, we return the symbol
            return symbol
        else:  # if not we remain in the same state
            return state

    fa = automata(my_special_transition)
    symbols = ["a", "b", "c", "ab", "abc", "abdc", "abcd"]
    assert list(fa("a", symbols)) == ["a", "a", "a", "ab", "abc", "abc", "abcd"]
