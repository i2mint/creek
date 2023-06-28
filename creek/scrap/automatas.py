"""Automatas (finite state machines etc.)"""

from typing import Any, TypeVar, Mapping, Iterable, Callable, Tuple

State = TypeVar('State')
Symbol = TypeVar('Symbol')
Automata = Callable[[State, Iterable[Symbol]], Iterable[State]]

TransitionFunc = Callable[[State, Symbol], State]
AutomataFactory = Callable[[TransitionFunc], Automata]


def test_automata(automata):
    # A automata that even number of 0s and 1s
    transition_func = mapping_to_transition_func(
        {
            ('even', 0): 'odd',
            ('odd', 1): 'even',
        },
        strict=False  # so that all (state, symbol) combinations not listed have no effect
    )
    fa = automata(transition_func)
    symbols = [0, 1, 0, 1, 1]
    it = fa('even', symbols)
    next(it) == 'even'  # reads 0, stays on state 'even'
    next(it) == 'odd'   # reads 1, applies the ('even', 0): 'odd' transition rule
    next(it) == 'odd'   # reads 0, stays on state 'odd'
    next(it) == 'even'  # reads 1, applies the ('odd', 1): 'even' transition rule
    next(it) == 'odd'   # reads 1, applies the ('even', 0): 'odd' transition rule

    # We can feed the automata with any iterable, and gather the "trace" of all the states 
    # it went through like follows. Notice that we can really put any symbol in the 
    # iterable, not just 0s and 1s. The automata will just ignore the symbols (remainingin 
    # in the same state) if there is no transition for the current state and the symbol.
    assert list(fa('even', [0, 1, 0, 42, 'not_even_a_number', 1])) == [
        'odd', 'even', 'odd', 'odd', 'odd', 'even'
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
    symbols = [
        'a', 'b', 'c', 'ab', 'abc', 'abdc', 'abcd'
    ]
    assert list(fa('a', symbols)) == [
        'a', 'a', 'a', 'ab', 'abc', 'abc', 'abcd'
    ]


def mapping_to_transition_func(
        mapping: Mapping[Tuple[State, Symbol], State], 
        strict: bool = True
) -> TransitionFunc:
    """
    Helper to make a transition function from a mapping of (state, symbol)->state.
    """
    if strict:
        def transition_func(state: State, symbol: Symbol) -> State:
            return mapping[(state, symbol)]
    else:
        def transition_func(state: State, symbol: Symbol) -> State:
            return mapping.get((state, symbol), state)
    return transition_func


# functional version
def _basic_automata(
        transition_func: TransitionFunc, 
        initial_state: State, 
        symbols: Iterable[Symbol]
) -> State:
    state = initial_state
    for symbol in symbols:
        # Note: if the (state, symbol) combination is not in the transitions
        #     mapping, the state is left unchanged.
        state = transition_func(state, symbol)
        yield state

from functools import partial

automata: AutomataFactory = (
    lambda transition_func: partial(_basic_automata, transition_func)
)
# # NerdNote: Could do it like this too
# basic_automata: AutomataFactory = partial(partial, _basic_automata)

test_automata(automata)


from dataclasses import dataclass

@dataclass
class BasicAutomata:
    transition_func: TransitionFunc
    _current_state: State = None

    def __call__(self, state: State, symbols: Iterable[Symbol]) -> State:
        self._current_state = state
        for symbol in symbols:
            yield self.transition(symbol)
   
    def transition(self, symbol: Symbol) -> State:
        self._current_state = self.transition_func(self._current_state, symbol)
        return self._current_state
    
test_automata(BasicAutomata)

