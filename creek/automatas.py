"""Automatas (finite state machines etc.)"""

from typing import Union, TypeVar, Mapping, Iterable, Callable, Tuple
from functools import partial
from dataclasses import dataclass

State = TypeVar('State')
Symbol = TypeVar('Symbol')
Automata = Callable[[State, Iterable[Symbol]], Iterable[State]]

TransitionFunc = Callable[[State, Symbol], State]
AutomataFactory = Callable[[TransitionFunc], Automata]


def mapping_to_transition_func(
    mapping: Mapping[Tuple[State, Symbol], State], strict: bool = True
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


StateMapper = Union[Callable[[State], State], Mapping[State, State]]


@dataclass
class MappingTransitionFunc:
    mapping: Mapping[Tuple[State, Symbol], State]
    strict: bool = True

    def __call__(self, state: State, symbol: Symbol) -> State:
        if self.strict:
            return self.mapping[(state, symbol)]
        else:
            return self.mapping.get((state, symbol), state)

    def map_states(self, state_mapper: StateMapper) -> 'MappingTransitionFunc':
        """Return a new MappingTransitionFunc with the same mapping but with
        state_mapper applied to the states."""
        if isinstance(state_mapper, Mapping):
            state_mapper = state_mapper.get
        new_mapping = {
            (state_mapper(state), symbol): state_mapper(state)
            for (state, symbol), state in self.mapping.items()
        }
        return MappingTransitionFunc(new_mapping, self.strict)


# functional version
def _basic_automata(
    transition_func: TransitionFunc, state: State, symbols: Iterable[Symbol]
) -> State:
    for symbol in symbols:
        # Note: if the (state, symbol) combination is not in the transitions
        #     mapping, the state is left unchanged.
        state = transition_func(state, symbol)
        yield state


basic_automata: AutomataFactory
BasicAutomata: AutomataFactory


# # NerdNote: Could do it like this too
# basic_automata: AutomataFactory = partial(partial, _basic_automata)
def basic_automata(transition_func: TransitionFunc, state: State) -> Automata:
    return partial(_basic_automata, transition_func, state)


@dataclass
class BasicAutomata:
    transition_func: TransitionFunc
    state: State = None

    def __post_init__(self):
        self._initial_state = self.state

    def __call__(self, state: State, symbols: Iterable[Symbol]) -> State:
        self.state = state
        for symbol in symbols:
            yield self.transition(symbol)

    def transition(self, symbol: Symbol) -> State:
        self.state = self.transition_func(self.state, symbol)
        return self.state

    def reset(self):
        """Reset state to initial state and return self."""
        self.state = self._initial_state
        return self


automata: AutomataFactory = basic_automata  # back-compatibility alias
