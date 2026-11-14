"""Edge-case matrix: inline automata run through the whole pipeline.

Every case must pass bounded verification against the original automaton,
and additionally satisfy one case-specific property.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from core.pipeline import PipelineResult, run_pipeline
from core.regex_ast import EMPTY, EPS, nullable
from core.solver import StepKind
from core.verifier import count_strings


def fa(
    states: list[str],
    alphabet: list[str],
    start: str,
    finals: list[str],
    moves: list[tuple[str, str, str]],
) -> dict[str, Any]:
    """Inline automaton in the JSON schema."""
    return {
        "name": "edge case",
        "states": states,
        "alphabet": alphabet,
        "start": start,
        "finals": finals,
        "transitions": [{"from": s, "symbol": x, "to": t} for s, x, t in moves],
    }


def kinds(r: PipelineResult) -> list[str]:
    return [p.kind for p in r.preprocessing]


def arden_count(r: PipelineResult) -> int:
    return sum(1 for s in r.solve.steps if s.kind is StepKind.ARDEN)


def mod6_dfa() -> dict[str, Any]:
    """Binary numbers divisible by 6: state rK means value mod 6 = K (6 states)."""
    states = [f"r{k}" for k in range(6)]
    moves = [(f"r{k}", b, f"r{(2 * k + int(b)) % 6}") for k in range(6) for b in "01"]
    return fa(states, ["0", "1"], "r0", ["r0"], moves)


Case = tuple[dict[str, Any], Callable[[PipelineResult], bool]]

CASES: dict[str, Case] = {
    "no final states -> empty set": (
        fa(["q0", "q1"], ["a"], "q0", [], [("q0", "a", "q1"), ("q1", "a", "q0")]),
        lambda r: r.solve.regex == EMPTY and r.regex_text == "∅",
    ),
    "start state final -> nullable": (
        fa(["q0", "q1"], ["a", "b"], "q0", ["q0"], [("q0", "a", "q1"), ("q1", "b", "q0")]),
        lambda r: nullable(r.solve.regex),
    ),
    "single final state, no transitions -> epsilon": (
        fa(["q0"], ["a"], "q0", ["q0"], []),
        lambda r: r.solve.regex == EPS and r.regex_text == "ε",
    ),
    "single non-final state -> empty set": (
        fa(["q0"], ["a"], "q0", [], []),
        lambda r: r.solve.regex == EMPTY,
    ),
    "self-loops on every state": (
        fa(["q0", "q1", "q2"], ["a", "b"], "q0", ["q2"],
           [("q0", "a", "q0"), ("q1", "a", "q1"), ("q2", "b", "q2"),
            ("q0", "b", "q1"), ("q1", "b", "q2")]),
        lambda r: arden_count(r) == 3,  # one Arden step per self-loop
    ),
    "unreachable state": (
        fa(["q0", "q1", "lost"], ["a"], "q0", ["q1"],
           [("q0", "a", "q1"), ("lost", "a", "q1")]),
        lambda r: kinds(r) == ["remove_unreachable"] and "lost" not in r.normalized.states,
    ),
    "dead trap state": (
        fa(["q0", "q1", "trap"], ["a", "b"], "q0", ["q1"],
           [("q0", "a", "q1"), ("q0", "b", "trap"), ("q1", "b", "trap"),
            ("trap", "a", "trap"), ("trap", "b", "trap")]),
        lambda r: kinds(r) == ["remove_dead"] and r.normalized.states == ("q0", "q1"),
    ),
    "multiple final states": (
        fa(["q0", "q1", "q2"], ["a", "b"], "q0", ["q1", "q2"],
           [("q0", "a", "q1"), ("q0", "b", "q2"), ("q1", "a", "q1"), ("q2", "b", "q2")]),
        lambda r: next(
            s for s in r.solve.steps if s.kind is StepKind.COMBINE_FINALS
        ).affected_states == ("q1", "q2"),
    ),
    "NFA with nondeterminism": (
        fa(["q0", "q1", "q2"], ["a", "b"], "q0", ["q2"],
           [("q0", "a", "q0"), ("q0", "b", "q0"), ("q0", "a", "q1"), ("q1", "b", "q2")]),
        lambda r: not r.original.is_deterministic() and r.original.accepts("bab"),
    ),
    "epsilon-NFA with an epsilon-cycle": (
        fa(["q0", "q1", "q2"], ["a", "b"], "q0", ["q2"],
           [("q0", "ε", "q1"), ("q1", "eps", "q0"), ("q0", "a", "q0"),
            ("q1", "b", "q1"), ("q1", "a", "q2")]),
        lambda r: kinds(r)[0] == "remove_epsilon" and not r.normalized.has_epsilon(),
    ),
    "alphabet of 3 symbols": (
        fa(["q0", "q1"], ["a", "b", "c"], "q0", ["q1"],
           [("q0", "a", "q1"), ("q1", "b", "q0"), ("q1", "c", "q1")]),
        lambda r: r.verification.strings_checked == count_strings(3, 8),
    ),
    "6-state DFA, default order, no blow-up": (
        mod6_dfa(),
        lambda r: len(r.normalized.states) == 6 and r.original.accepts("110")  # 6
        and not r.original.accepts("111"),  # 7
    ),
}


@pytest.mark.parametrize(("spec", "expected"), list(CASES.values()), ids=list(CASES))
def test_edge_case(spec: dict[str, Any], expected: Callable[[PipelineResult], bool]) -> None:
    result = run_pipeline(spec)
    v = result.verification
    assert v.passed, (result.regex_text, v.counterexample)
    assert not v.truncated
    assert expected(result), result.regex_text
