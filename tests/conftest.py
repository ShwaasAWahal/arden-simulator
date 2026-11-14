"""Shared test helpers: hypothesis strategies used by several test modules."""

from __future__ import annotations

from hypothesis import strategies as st

from core.automaton import Automaton, parse_automaton


@st.composite
def random_nfas(draw: st.DrawFn) -> Automaton:
    """Random NFAs: 2-5 states, alphabet {a, b}, transitions including some ε."""
    n = draw(st.integers(min_value=2, max_value=5))
    states = [f"q{i}" for i in range(n)]
    state = st.sampled_from(states)
    transitions = draw(
        st.lists(
            st.tuples(state, st.sampled_from(["a", "b", "ε", "eps"]), state),
            max_size=3 * n,
        )
    )
    finals = draw(st.lists(state, unique=True))
    start = draw(state)
    return parse_automaton(
        {
            "name": "random",
            "states": states,
            "alphabet": ["a", "b"],
            "start": start,
            "finals": finals,
            "transitions": [{"from": s, "symbol": x, "to": t} for s, x, t in transitions],
        }
    )
