"""Tests for core.equations: equation model, substitution, builder, printers."""

from __future__ import annotations

from pathlib import Path

import pytest

from core import ArdenPreconditionError, SolverError
from core.automaton import load_automaton, normalize, parse_automaton
from core.equations import Equation, EquationSystem, build_equations, state_to_latex
from core.regex_ast import EMPTY, EPS, concat, nullable, star, sym, union

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_FILES = sorted(p.name for p in EXAMPLES_DIR.glob("*.json"))

a, b = sym("a"), sym("b")
zero, one = sym("0"), sym("1")


def build(name: str) -> EquationSystem:
    return build_equations(load_automaton(EXAMPLES_DIR / name))


# ---------------------------------------------------------------------------
# Builder on examples
# ---------------------------------------------------------------------------


def test_ends_with_01_equations() -> None:
    system = build("ends_with_01.json")
    # Incoming transitions, written out by hand:
    #   q0 <- q0 on 1, q2 on 1   (+ ε, start state)
    #   q1 <- q0 on 0, q1 on 0, q2 on 0
    #   q2 <- q1 on 1
    expected = (
        Equation("q0", (("q0", one), ("q2", one)), EPS),
        Equation("q1", (("q0", zero), ("q1", zero), ("q2", zero)), EMPTY),
        Equation("q2", (("q1", one),), EMPTY),
    )
    assert system.equations == expected
    assert system.state_order == ("q0", "q1", "q2")
    assert system.start == "q0" and system.finals == frozenset({"q2"})
    assert system.to_text() == (
        "q0 = q0·1 + q2·1 + ε\n"
        "q1 = q0·0 + q1·0 + q2·0\n"
        "q2 = q1·1"
    )


def test_even_zeros_start_has_epsilon_and_loop() -> None:
    system = build("even_zeros.json")
    start = system.get("even")
    assert start.const == EPS
    assert start.has_self_reference()
    assert start.self_coeff() == one
    assert start.coeff("odd") == zero
    assert system.to_text() == "even = even·1 + odd·0 + ε\nodd = even·0 + odd·1"


def test_parallel_transitions_collect_into_union() -> None:
    aut = parse_automaton(
        {
            "states": ["q0", "q1"],
            "alphabet": ["a", "b"],
            "start": "q0",
            "finals": ["q1"],
            "transitions": [
                {"from": "q0", "symbol": "a", "to": "q1"},
                {"from": "q0", "symbol": "b", "to": "q1"},
            ],
        }
    )
    system = build_equations(aut)
    assert system.get("q1").coeff("q0") == union(a, b)
    assert system.get("q1").to_text() == "q1 = q0·(a+b)"
    assert system.get("q0").to_text() == "q0 = ε"


def test_state_without_incoming_gets_empty_equation() -> None:
    # q3 is unreachable and has no incoming transitions (not normalized on purpose).
    system = build("unreachable_dead.json")
    eq = system.get("q3")
    assert eq.const == EMPTY and eq.is_closed()
    assert eq.to_text() == "q3 = ∅"
    assert eq.to_latex() == r"q_{3} = \varnothing"


def test_builder_rejects_epsilon_moves() -> None:
    aut = load_automaton(EXAMPLES_DIR / "epsilon_nfa.json")
    with pytest.raises(ArdenPreconditionError, match="normalize"):
        build_equations(aut)
    # ...but works after normalization.
    normalized, _ = normalize(aut)
    assert build_equations(normalized).get("q0").const == EPS


# ---------------------------------------------------------------------------
# Like-term collection and equality
# ---------------------------------------------------------------------------


def test_like_terms_collected() -> None:
    eq = Equation("q1", (("q0", a), ("q0", b)))
    assert eq.coeffs == (("q0", union(a, b)),)
    assert eq.to_text() == "q1 = q0·(a+b)"


def test_empty_coefficients_dropped_and_mapping_accepted() -> None:
    eq = Equation("q1", {"q0": EMPTY, "q2": a})
    assert eq.variables() == frozenset({"q2"})
    assert eq.coeff("q0") == EMPTY


def test_equation_equality_independent_of_term_order() -> None:
    e1 = Equation("q2", (("q0", a), ("q1", b)), EPS)
    e2 = Equation("q2", (("q1", b), ("q0", a)), EPS)
    assert e1 == e2 and hash(e1) == hash(e2)


# ---------------------------------------------------------------------------
# substitute()
# ---------------------------------------------------------------------------


def test_substitute_creates_self_reference() -> None:
    # q2 = q1·1,  q1 = q0·0 + q1·0 + q2·0
    # => q2 = q0·01 + q1·01 + q2·01   (q2 now refers to itself)
    q2 = Equation("q2", (("q1", one),))
    q1 = Equation("q1", (("q0", zero), ("q1", zero), ("q2", zero)))
    assert not q2.has_self_reference()
    result = q2.substitute("q1", q1)
    assert result == Equation(
        "q2",
        (("q0", concat(zero, one)), ("q1", concat(zero, one)), ("q2", concat(zero, one))),
    )
    assert result.has_self_reference()
    assert result.to_text() == "q2 = q0·01 + q1·01 + q2·01"


def test_substitute_with_constant() -> None:
    # q1 = q0·a,  q0 = q1·b + ε   =>  q1 = q1·ba + a
    q1 = Equation("q1", (("q0", a),))
    q0 = Equation("q0", (("q1", b),), EPS)
    result = q1.substitute("q0", q0)
    assert result == Equation("q1", (("q1", concat(b, a)),), a)
    assert result.to_text() == "q1 = q1·ba + a"
    assert result.self_coeff() == concat(b, a)


def test_substitute_closed_form_and_like_term_merge() -> None:
    # q2 = q0·a + q1·b,  q1 = q0·b + a*   =>  q2 = q0·(a+bb) + a*b
    q2 = Equation("q2", (("q0", a), ("q1", b)))
    q1 = Equation("q1", (("q0", b),), star(a))
    result = q2.substitute("q1", q1)
    assert result.coeff("q0") == union(a, concat(b, b))
    assert result.const == concat(star(a), b)
    assert result.to_text() == "q2 = q0·(a+bb) + a*b"


def test_substitute_simplifies_coefficients() -> None:
    # q2 = q1·a*,  q1 = q0·a*   =>  q2 = q0·a*a* = q0·a*
    result = Equation("q2", (("q1", star(a)),)).substitute(
        "q1", Equation("q1", (("q0", star(a)),))
    )
    assert result.to_text() == "q2 = q0·a*"


def test_substitute_absent_variable_is_noop() -> None:
    eq = Equation("q2", (("q0", a),), EPS)
    assert eq.substitute("q1", Equation("q1", (), b)) is eq


def test_substitute_rejects_mismatched_equation() -> None:
    with pytest.raises(ValueError):
        Equation("q2", (("q1", a),)).substitute("q1", Equation("q0", (), EPS))


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def test_self_coeff_and_is_closed() -> None:
    loop = Equation("q0", (("q0", a), ("q1", b)), EPS)
    assert loop.self_coeff() == a
    assert loop.has_self_reference()
    assert not loop.is_closed()
    assert loop.variables() == frozenset({"q0", "q1"})

    no_loop = Equation("q1", (("q0", a),))
    assert no_loop.self_coeff() == EMPTY
    assert not no_loop.has_self_reference()
    assert not no_loop.is_closed()

    closed = Equation("q0", (), concat(a, star(b)))
    assert closed.is_closed()
    assert closed.variables() == frozenset()
    assert closed.to_text() == "q0 = ab*"


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------


def test_text_and_latex_printers() -> None:
    eq = Equation("q1", (("q1", one), ("q0", zero)), EPS)
    assert eq.to_text() == "q1 = q0·0 + q1·1 + ε"
    assert str(eq) == eq.to_text()
    assert eq.to_latex() == r"q_{1} = q_{0}0 + q_{1}1 + \varepsilon"


def test_term_order_follows_state_order() -> None:
    eq = Equation("q1", (("q0", a), ("q10", b), ("q2", a)))
    assert eq.to_text() == "q1 = q0·a + q2·a + q10·b"  # natural order by default
    assert eq.to_text(("q10", "q2", "q0", "q1")) == "q1 = q10·b + q2·a + q0·a"


def test_constant_union_printed_as_separate_summands() -> None:
    eq = Equation("q1", (("q0", a),), union(b, EPS))
    assert eq.to_text() == "q1 = q0·a + ε + b"
    assert eq.to_latex() == r"q_{1} = q_{0}a + \varepsilon + b"


@pytest.mark.parametrize(
    ("name", "latex"),
    [
        ("q12", "q_{12}"),
        ("p0", "p_{0}"),
        ("A", r"\mathit{A}"),
        ("start", r"\mathit{start}"),
        ("q_x", r"\mathit{q\_x}"),
    ],
)
def test_state_to_latex(name: str, latex: str) -> None:
    assert state_to_latex(name) == latex


def test_non_q_state_names() -> None:
    system = build("even_zeros.json")
    assert system.get("odd").to_latex() == r"\mathit{odd} = \mathit{even}0 + \mathit{odd}1"
    eq = Equation("A", (("start", union(a, b)),))
    assert eq.to_text() == "A = start·(a+b)"
    assert eq.to_latex() == r"\mathit{A} = \mathit{start}(a+b)"


def test_system_latex_aligned() -> None:
    system = build("ends_with_01.json")
    assert system.to_latex() == (
        "\\begin{aligned}\n"
        "q_{0} &= q_{0}1 + q_{2}1 + \\varepsilon \\\\\n"
        "q_{1} &= q_{0}0 + q_{1}0 + q_{2}0 \\\\\n"
        "q_{2} &= q_{1}1\n"
        "\\end{aligned}"
    )


# ---------------------------------------------------------------------------
# EquationSystem get / replace
# ---------------------------------------------------------------------------


def test_system_get_and_replace() -> None:
    system = build("ends_with_01.json")
    new_q2 = Equation("q2", (), concat(zero, one))
    updated = system.replace(new_q2)
    assert updated.get("q2") == new_q2
    assert system.get("q2") != new_q2  # original untouched (immutable)
    assert updated.state_order == system.state_order
    with pytest.raises(SolverError):
        system.get("nope")
    with pytest.raises(SolverError):
        system.replace(Equation("nope", (), EPS))


def test_system_rejects_mismatched_order() -> None:
    with pytest.raises(ValueError):
        EquationSystem((Equation("q1"),), ("q0",), "q0", frozenset())


# ---------------------------------------------------------------------------
# Language sanity: Arden's precondition will hold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_coefficients_are_not_nullable(filename: str) -> None:
    # After normalize() the automaton has no ε-moves, so every builder
    # coefficient is a union of alphabet symbols: it denotes only strings of
    # length 1 and is therefore not nullable. Substitution only ever forms
    # C_k·C with C non-nullable, which stays non-nullable, so when an equation
    # later reaches the form q = Q + q·P, its P is never nullable and Arden's
    # theorem (R = Q + RP ⇒ R = QP*, P not nullable) applies with a unique solution.
    normalized, _ = normalize(load_automaton(EXAMPLES_DIR / filename))
    system = build_equations(normalized)
    for eq in system.equations:
        for var, coeff in eq.coeffs:
            assert not nullable(coeff), f"{filename}: {eq.lhs} has nullable coeff on {var}"
    assert system.get(system.start).const == EPS
