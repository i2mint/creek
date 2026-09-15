"""Automatas (finite state machines): step a state through a stream of symbols.

Main entry points:

- ``mapping_to_transition_func``: a transition function from a ``{(state, symbol): next_state}`` mapping
- ``basic_automata``: ``basic_automata(transition_func, state)(symbols)`` yields the state after each symbol
- ``BasicAutomata``: the same as a stateful, resettable object

>>> from creek.automatas import mapping_to_transition_func, basic_automata
>>> step = mapping_to_transition_func({('a', 1): 'b', ('b', 1): 'a'})
>>> list(basic_automata(step, 'a')([1, 1, 1]))
['b', 'a', 'b']
"""

from typing import Union, TypeVar, Tuple
from collections.abc import Mapping, Iterable, Callable
from functools import partial
from dataclasses import dataclass

State = TypeVar("State")
Symbol = TypeVar("Symbol")
Automata = Callable[[State, Iterable[Symbol]], Iterable[State]]

TransitionFunc = Callable[[State, Symbol], State]
AutomataFactory = Callable[[TransitionFunc], Automata]


def mapping_to_transition_func(
    mapping: Mapping[tuple[State, Symbol], State], strict: bool = True
) -> TransitionFunc:
    """Make a transition function from a ``{(state, symbol): next_state}`` mapping.

    Args:
        mapping: The transitions.
        strict: If true (the default), an unmapped ``(state, symbol)`` pair is a
            lookup error (``KeyError`` for a ``dict``); if false, the state is left
            unchanged.
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
    """Class form of ``mapping_to_transition_func``: a callable ``(state, symbol) -> next_state``."""

    mapping: Mapping[tuple[State, Symbol], State]
    strict: bool = True

    def __call__(self, state: State, symbol: Symbol) -> State:
        """Return the next state for ``(state, symbol)``."""
        if self.strict:
            return self.mapping[(state, symbol)]
        else:
            return self.mapping.get((state, symbol), state)

    def map_states(self, state_mapper: StateMapper) -> "MappingTransitionFunc":
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
    """Make an automata: ``basic_automata(f, state)(symbols)`` yields the state after each symbol."""
    return partial(_basic_automata, transition_func, state)


@dataclass
class BasicAutomata:
    """Stateful automata: ``automata(state, symbols)`` yields the state after each symbol; ``reset`` restores the initial one."""

    transition_func: TransitionFunc
    state: State = None

    def __post_init__(self):
        self._initial_state = self.state

    def __call__(self, state: State, symbols: Iterable[Symbol]) -> State:
        """Set the state to ``state``, then yield the state after each of ``symbols``."""
        self.state = state
        for symbol in symbols:
            yield self.transition(symbol)

    def transition(self, symbol: Symbol) -> State:
        """Apply one ``symbol`` to the current state and return the new state."""
        self.state = self.transition_func(self.state, symbol)
        return self.state

    def reset(self):
        """Reset state to initial state and return self."""
        self.state = self._initial_state
        return self


automata: AutomataFactory = basic_automata  # back-compatibility alias
