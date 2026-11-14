"""Tests for core.solver: Arden elimination, orders, trace structure, safety."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from hypothesis import given, settings

import core.solver as solver_mod
from core import ArdenPreconditionError, SolverError
from core.automaton import Automaton, load_automaton, normalize
from core.equations import Equation, EquationSystem, build_equations
from core.regex_ast import (
    EMPTY,
    EPS,
    concat,
    nullable,
    star,
    sym,
    to_text,
    union,
)
from core.solver import (
    SolveResult,
    StepKind,
    apply_arden,
    default_order,
    solve,
    solve_all_orders,
)
from core.verifier import verify
from tests.conftest import random_nfas

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_FILES = sorted(p.name for p in EXAMPLES_DIR.glob("*.json"))

a, b = sym("a"), sym("b")


def prepared(name: str) -> tuple[Automaton, EquationSystem]:
    """Load an example, normalize it and build its equation system."""
    aut = load_automaton(EXAMPLES_DIR / name)
    normalized, _ = normalize(aut)
    return aut, build_equations(normalized)


def assert_same_language(aut: Automaton, result: SolveResult, n: int) -> None:
    """The solved RE agrees with the original automaton on all strings up to length n."""
    v = verify(aut, result.regex, max_length=n)
    assert v.passed, (
        f"order {result.order}: RE {to_text(result.regex)} disagrees on "
        f"{v.counterexample!r} (FA={v.fa_accepts_counterexample}, RE={v.re_accepts_counterexample})"
    )
    assert not v.truncated and v.max_length == n


# ---------------------------------------------------------------------------
# Correctness on the examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_default_order_matches_automaton(filename: str) -> None:
    aut, system = prepared(filename)
    assert_same_language(aut, solve(system), 8)


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_every_order_matches_automaton(filename: str) -> None:
    aut, system = prepared(filename)
    assert len(system.state_order) <= 4
    results = solve_all_orders(system)
    n = len(system.state_order)
    assert len(results) == min(24, len(list(itertools.permutations(range(n)))))
    assert len({r.order for r in results}) == len(results)  # all distinct orders
    for r in results:
        assert_same_language(aut, r, 8)


def test_single_state_loop_exact_text() -> None:
    _, system = prepared("single_state_loop.json")
    assert to_text(solve(system).regex) == "(a+b)*"


def test_no_finals_is_empty() -> None:
    _, system = prepared("no_finals.json")
    result = solve(system)
    assert result.regex == EMPTY
    combine = [s for s in result.steps if s.kind is StepKind.COMBINE_FINALS][0]
    assert "no final states" in combine.explanation


def test_even_zeros_is_nullable() -> None:
    _, system = prepared("even_zeros.json")
    assert nullable(solve(system).regex)


def test_closed_forms_cover_every_state() -> None:
    aut, system = prepared("ends_with_01.json")
    result = solve(system)
    assert set(result.closed_forms) == set(system.state_order)
    # q2 is the only final state, so the RE is its closed form (simplified).
    assert verify(aut, result.closed_forms["q2"], max_length=8).passed


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def test_default_order_reverse_with_start_last() -> None:
    _, system = prepared("ends_with_01.json")
    assert default_order(system) == ("q2", "q1", "q0")
    custom = EquationSystem(
        (Equation("x"), Equation("s", (), EPS), Equation("y")),
        ("x", "s", "y"),
        "s",
        frozenset(),
    )
    assert default_order(custom) == ("y", "x", "s")


@pytest.mark.parametrize(
    ("order", "fragment"),
    [
        (("q0", "q1"), "missing state(s) 'q2'"),
        (("q0", "q1", "q2", "q2"), "repeated state(s) 'q2'"),
        (("q0", "q1", "q9"), "unknown state(s) 'q9'"),
        ((), "missing state(s) 'q0', 'q1', 'q2'"),
    ],
)
def test_invalid_order_raises(order: tuple[str, ...], fragment: str) -> None:
    _, system = prepared("ends_with_01.json")
    with pytest.raises(SolverError) as info:
        solve(system, order)
    assert "invalid elimination order" in str(info.value)
    assert fragment in str(info.value)


def test_custom_order_is_used() -> None:
    aut, system = prepared("ends_with_01.json")
    result = solve(system, ["q0", "q1", "q2"])
    assert result.order == ("q0", "q1", "q2")
    assert_same_language(aut, result, 8)


def test_solve_all_orders_limit_and_size_check() -> None:
    _, system = prepared("ends_with_01.json")
    results = solve_all_orders(system, limit=2)
    assert len(results) == 2
    assert results[0].order == default_order(system)
    five = EquationSystem(
        tuple(Equation(f"q{i}") for i in range(5)),
        tuple(f"q{i}" for i in range(5)),
        "q0",
        frozenset(),
    )
    with pytest.raises(SolverError, match="at most 4 states"):
        solve_all_orders(five)


# ---------------------------------------------------------------------------
# Trace structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_trace_structure(filename: str) -> None:
    _, system = prepared(filename)
    result = solve(system)
    steps = result.steps
    assert [s.index for s in steps] == list(range(len(steps)))
    assert steps[0].kind is StepKind.INITIAL and steps[0].phase == "setup"
    assert steps[0].system_before == system
    assert steps[-1].kind is StepKind.RESULT
    assert steps[-2].kind is StepKind.COMBINE_FINALS
    for prev, nxt in zip(steps, steps[1:]):
        assert prev.system_after == nxt.system_before
    for s in steps:
        assert s.explanation and s.explanation.endswith(".")
        if s.kind in (StepKind.ARDEN, StepKind.SUBSTITUTE):
            assert s.phase == "elimination" and s.focus_state is not None
            assert s.affected_states and s.rule_latex
        if s.kind is StepKind.BACK_SUBSTITUTE:
            assert s.phase == "back_substitution"
    assert all(eq.is_closed() for eq in result.final_system.equations)
    if _has_cycle(system):  # any loop in the automaton forces an Arden step
        assert any(s.kind is StepKind.ARDEN for s in steps)
    # phases appear in non-decreasing order
    rank = {"setup": 0, "elimination": 1, "back_substitution": 2, "result": 3}
    assert [rank[s.phase] for s in steps] == sorted(rank[s.phase] for s in steps)


def _has_cycle(system: EquationSystem) -> bool:
    """Does the transition graph (read off the equations) contain a cycle?"""
    succ = {s: set() for s in system.state_order}
    for eq in system.equations:
        for v in eq.variables():
            succ[v].add(eq.lhs)

    def reach(x: str, target: str, seen: set[str]) -> bool:
        for y in succ[x]:
            if y == target:
                return True
            if y not in seen:
                seen.add(y)
                if reach(y, target, seen):
                    return True
        return False

    return any(reach(s, s, set()) for s in system.state_order)


def test_arden_step_content() -> None:
    _, system = prepared("ends_with_01.json")
    arden_steps = [s for s in solve(system).steps if s.kind is StepKind.ARDEN]
    assert arden_steps
    first = arden_steps[0]
    assert "references itself with P =" in first.explanation
    assert "Arden's Theorem" in first.explanation
    assert first.rule_latex and r"\Rightarrow" in first.rule_latex
    assert first.affected_states == (first.focus_state,)
    assert not first.system_after.get(first.focus_state).has_self_reference()


def test_solve_is_pure() -> None:
    _, system = prepared("contains_aa.json")
    snapshot = system.to_text()
    r1 = solve(system)
    r2 = solve(system)
    assert system.to_text() == snapshot
    assert r1.regex == r2.regex and r1.steps == r2.steps


# ---------------------------------------------------------------------------
# Arden precondition and safety guard
# ---------------------------------------------------------------------------


def test_apply_arden_basic() -> None:
    # q0 = q1·b + ε + q0·a  =>  q0 = q1·ba* + a*
    eq = Equation("q0", (("q0", a), ("q1", b)), EPS)
    solved = apply_arden(eq)
    assert solved == Equation("q0", (("q1", concat(b, star(a))),), star(a))
    assert apply_arden(solved) is solved  # no self-term: unchanged


def test_nullable_self_coefficient_raises() -> None:
    eq = Equation("q0", (("q0", union(EPS, a)),), EPS)
    with pytest.raises(ArdenPreconditionError, match="nullable"):
        apply_arden(eq)
    system = EquationSystem((eq,), ("q0",), "q0", frozenset({"q0"}))
    with pytest.raises(ArdenPreconditionError, match="normalize"):
        solve(system)


def test_expression_size_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    _, system = prepared("ends_with_01.json")
    monkeypatch.setattr(solver_mod, "MAX_COEFF_SIZE", 3)
    with pytest.raises(SolverError, match="expression too large"):
        solve(system)


# ---------------------------------------------------------------------------
# Property: random NFAs
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(random_nfas())
def test_solve_random_nfas(aut: Automaton) -> None:
    normalized, _ = normalize(aut)
    result = solve(build_equations(normalized))
    v = verify(aut, result.regex, max_length=6)
    assert v.passed, (to_text(result.regex), v.counterexample)
