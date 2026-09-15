# creek.automatas

Automatas (finite state machines): step a state through a stream of symbols.

Main entry points:

- `mapping_to_transition_func`: a transition function from a `{(state, symbol): next_state}` mapping
- `basic_automata`: `basic_automata(transition_func, state)(symbols)` yields the state after each symbol
- `BasicAutomata`: the same as a stateful, resettable object

```pycon
>>> from creek.automatas import mapping_to_transition_func, basic_automata
>>> step = mapping_to_transition_func({('a', 1): 'b', ('b', 1): 'a'})
>>> list(basic_automata(step, 'a')([1, 1, 1]))
['b', 'a', 'b']
```

### Functions

| [`automata`](#creek.automatas.automata)(transition_func, state)              | Make an automata: `basic_automata(f, state)(symbols)` yields the state after each symbol.   |
|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`basic_automata`](#creek.automatas.basic_automata)(transition_func, state)        | Make an automata: `basic_automata(f, state)(symbols)` yields the state after each symbol.   |
| [`mapping_to_transition_func`](#creek.automatas.mapping_to_transition_func)(mapping[, strict]) | Make a transition function from a `{(state, symbol): next_state}` mapping.                  |

### Classes

| [`BasicAutomata`](#creek.automatas.BasicAutomata)(transition_func[, state])   | Stateful automata: `automata(state, symbols)` yields the state after each symbol; `reset` restores the initial one.   |
|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| [`MappingTransitionFunc`](#creek.automatas.MappingTransitionFunc)(mapping[, strict])  | Class form of `mapping_to_transition_func`: a callable `(state, symbol) -> next_state`.                               |

### *class* creek.automatas.BasicAutomata(transition_func, state=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Stateful automata: `automata(state, symbols)` yields the state after each symbol; `reset` restores the initial one.

#### reset()

Reset state to initial state and return self.

#### transition(symbol)

Apply one `symbol` to the current state and return the new state.

* **Return type:**
  [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`State`)

### *class* creek.automatas.MappingTransitionFunc(mapping, strict=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Class form of `mapping_to_transition_func`: a callable `(state, symbol) -> next_state`.

#### map_states(state_mapper)

Return a new MappingTransitionFunc with the same mapping but with
state_mapper applied to the states.

* **Return type:**
  [`MappingTransitionFunc`](#creek.automatas.MappingTransitionFunc)

### creek.automatas.automata(transition_func, state)

Make an automata: `basic_automata(f, state)(symbols)` yields the state after each symbol.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`State`), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Symbol`)]], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`State`)]]

### creek.automatas.basic_automata(transition_func, state)

Make an automata: `basic_automata(f, state)(symbols)` yields the state after each symbol.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`State`), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Symbol`)]], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`State`)]]

### creek.automatas.mapping_to_transition_func(mapping, strict=True)

Make a transition function from a `{(state, symbol): next_state}` mapping.

* **Parameters:**
  * **mapping** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`State`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Symbol`)], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`State`)]) – The transitions.
  * **strict** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If true (the default), an unmapped `(state, symbol)` pair is a
    lookup error (`KeyError` for a `dict`); if false, the state is left
    unchanged.
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`State`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Symbol`)], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`State`)]
