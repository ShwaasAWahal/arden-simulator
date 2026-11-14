"""Arden's-theorem solver.

Responsibilities: Arden + substitution elimination, elimination-order
handling, back-substitution, and TraceStep events.

Algorithm (CLAUDE.md, "Applying it to an FA", step 2 and 3):

1. **Forward elimination.** For each state ``v`` in the elimination order:
   if its equation has the form ``v = Q + v·P``, apply Arden's theorem
   (``v = Q·P*``, P not nullable); then substitute ``v``'s equation into every
   later (pending) equation that mentions ``v``. After this phase each
   equation mentions only states *later* in the order, so the last one is closed.
2. **Back-substitution.** In reverse order, substitute the closed forms of all
   later states into each equation, making every equation closed.
3. **Combine finals.** The RE is the union of the closed forms of the final
   states (``∅`` if there are none).

Every change is recorded as a :class:`TraceStep` for the step-by-step UI.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Literal, Mapping, Sequence

from core import ArdenPreconditionError, SolverError
from core.equations import Equation, EquationSystem, state_to_latex
from core.regex_ast import (
    EMPTY,
    Regex,
    concat,
    nullable,
    simplify,
    size,
    star,
    to_latex,
    to_text,
    union,
)

__all__ = [
    "MAX_COEFF_SIZE",
    "MAX_STATES_ALL_ORDERS",
    "StepKind",
    "Phase",
    "TraceStep",
    "ReadOnlyMap",
    "SolveResult",
    "apply_arden",
    "default_order",
    "solve",
    "solve_all_orders",
]

#: Largest allowed size() of any coefficient or constant during solving.
MAX_COEFF_SIZE = 5000
#: solve_all_orders() only enumerates permutations for systems up to this many states.
MAX_STATES_ALL_ORDERS = 4


class StepKind(str, Enum):
    """Kind of a trace step."""

    INITIAL = "initial"
    ARDEN = "arden"
    SUBSTITUTE = "substitute"
    BACK_SUBSTITUTE = "back_substitute"
    COMBINE_FINALS = "combine_finals"
    RESULT = "result"


Phase = Literal["setup", "elimination", "back_substitution", "result"]


@dataclass(frozen=True)
class TraceStep:
    """One event of the solution, with the equation system before and after it."""

    index: int
    kind: StepKind
    phase: Phase
    focus_state: str | None
    affected_states: tuple[str, ...]
    explanation: str
    rule_latex: str | None
    system_before: EquationSystem = field(repr=False)
    system_after: EquationSystem = field(repr=False)


class ReadOnlyMap(Mapping[str, Regex]):
    """A read-only, picklable ``str -> Regex`` mapping.

    ``types.MappingProxyType`` would also be read-only, but it cannot be
    pickled, which breaks caching (e.g. Streamlit's ``st.cache_data``).
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Regex]) -> None:
        """Copy ``data`` so later changes to the source cannot leak in."""
        self._data: dict[str, Regex] = dict(data)

    def __getitem__(self, key: str) -> Regex:
        """Look up the closed form of state ``key``."""
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over state names."""
        return iter(self._data)

    def __len__(self) -> int:
        """Number of states."""
        return len(self._data)

    def __repr__(self) -> str:
        """Readable representation, like a dict."""
        return f"ReadOnlyMap({self._data!r})"

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle support: rebuild from a plain dict."""
        return (ReadOnlyMap, (self._data,))


@dataclass(frozen=True)
class SolveResult:
    """Outcome of :func:`solve`.

    Attributes:
        regex: the simplified regular expression of the automaton's language.
        closed_forms: closed-form regex of every state (``state -> Regex``), as a
            read-only (but picklable) mapping so the frozen result cannot be mutated.
        order: the elimination order that was used.
        steps: the full trace, indices 0..n-1.
    """

    regex: Regex
    closed_forms: Mapping[str, Regex] = field(hash=False)
    order: tuple[str, ...]
    steps: tuple[TraceStep, ...]

    @property
    def final_system(self) -> EquationSystem:
        """The fully closed equation system after the last step."""
        return self.steps[-1].system_after


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _check_size(eq: Equation) -> Equation:
    """Raise SolverError if any coefficient/constant of ``eq`` exceeds MAX_COEFF_SIZE."""
    for r in (*(c for _, c in eq.coeffs), eq.const):
        if size(r) > MAX_COEFF_SIZE:
            raise SolverError(
                "expression too large; try a different elimination order or a smaller automaton"
            )
    return eq


def _without_self(eq: Equation) -> Equation:
    """The Q part of ``v = Q + v·P``, as an equation for ``v``."""
    return Equation(eq.lhs, tuple(t for t in eq.coeffs if t[0] != eq.lhs), eq.const)


def apply_arden(eq: Equation) -> Equation:
    """Applies Arden's theorem: R = Q + RP ⇒ R = QP*, P not nullable.

    In left-linear form, with ``v = Σ q_k·C_k + K + v·P``, every coefficient
    ``C_k`` becomes ``C_k·P*`` and the constant ``K`` becomes ``K·P*``; the
    self-term disappears. An equation without a self-term is returned as is.

    Raises:
        ArdenPreconditionError: if P is nullable (the solution would not be unique).
    """
    p = eq.self_coeff()
    if p == EMPTY:
        return eq
    if nullable(p):
        raise ArdenPreconditionError(
            f"cannot apply Arden's theorem to {eq.lhs}: P = {to_text(p)} is nullable "
            "(it contains ε), so R = Q + RP has no unique solution. This cannot happen "
            "for an automaton without ε-transitions; remove them first (the simulator's "
            "preprocessing, normalize(), does this)."
        )
    p_star = star(p)
    q = _without_self(eq)
    return _check_size(
        Equation(
            eq.lhs,
            tuple((v, concat(c, p_star)) for v, c in q.coeffs),
            concat(q.const, p_star),
        ).simplified()
    )


def default_order(system: EquationSystem) -> tuple[str, ...]:
    """Reverse declaration order with the start state moved to the very end.

    The start state is therefore solved last, as in most textbooks.
    """
    rev = [s for s in reversed(system.state_order) if s != system.start]
    return (*rev, system.start)


def _validate_order(system: EquationSystem, order: Sequence[str]) -> tuple[str, ...]:
    """Check that ``order`` is a permutation of the system's states."""
    order = tuple(order)
    states = set(system.state_order)
    seen: set[str] = set()
    dups: list[str] = []
    for s in order:
        if s in seen:
            dups.append(s)
        seen.add(s)
    unknown = [s for s in order if s not in states]
    missing = [s for s in system.state_order if s not in seen]
    problems = []
    if unknown:
        problems.append("unknown state(s) " + ", ".join(map(repr, unknown)))
    if dups:
        problems.append("repeated state(s) " + ", ".join(map(repr, dict.fromkeys(dups))))
    if missing:
        problems.append("missing state(s) " + ", ".join(map(repr, missing)))
    if problems:
        raise SolverError(
            "invalid elimination order: it must list every state exactly once ("
            + "; ".join(problems)
            + f"). States: {', '.join(system.state_order)}."
        )
    return order


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class _Trace:
    """Accumulates TraceSteps with consecutive indices."""

    def __init__(self) -> None:
        """Start with an empty trace."""
        self.steps: list[TraceStep] = []

    def emit(
        self,
        kind: StepKind,
        phase: Phase,
        focus: str | None,
        affected: Sequence[str],
        explanation: str,
        rule_latex: str | None,
        before: EquationSystem,
        after: EquationSystem,
    ) -> None:
        """Append a TraceStep whose index is the next consecutive number."""
        self.steps.append(
            TraceStep(
                len(self.steps), kind, phase, focus, tuple(affected),
                explanation, rule_latex, before, after,
            )
        )


def _arden_step(system: EquationSystem, v: str, trace: _Trace, phase: Phase) -> EquationSystem:
    """Apply Arden's theorem (R = Q + RP ⇒ R = QP*, P not nullable) to ``v``'s
    equation if it is self-referential, recording an ARDEN step."""
    order = system.state_order
    eq = system.get(v)
    if not eq.has_self_reference():
        return system
    p = eq.self_coeff()
    q = _without_self(eq)
    solved = apply_arden(eq)  # raises ArdenPreconditionError if P is nullable
    after = system.replace(solved)
    q_text = q.rhs_text(order)
    p_star = to_text(star(p))
    explanation = (
        f"{v} references itself with P = {to_text(p)}. "
        f"By Arden's Theorem (P is not nullable), {v} = ({q_text})·{p_star}, "
        f"which simplifies to {solved.to_text(order)}."
    )
    v_l = state_to_latex(v)
    rule = (
        f"{v_l} = Q + {v_l}P,\\; Q = {q.rhs_latex(order)},\\; P = {to_latex(p)} "
        f"\\;\\Rightarrow\\; {v_l} = {solved.rhs_latex(order)}"
    )
    trace.emit(StepKind.ARDEN, phase, v, (v,), explanation, rule, system, after)
    return after


def solve(system: EquationSystem, order: Sequence[str] | None = None) -> SolveResult:
    """Solve a left-linear equation system with Arden's theorem and substitution.

    Args:
        system: the system built by ``build_equations`` (not modified).
        order: elimination order, a permutation of all states; defaults to
            :func:`default_order`. Different orders give different-looking but
            language-equivalent regexes.

    Raises:
        SolverError: invalid order, or an expression grows beyond MAX_COEFF_SIZE.
        ArdenPreconditionError: a self-coefficient P is nullable.
    """
    order = default_order(system) if order is None else _validate_order(system, order)
    state_order = system.state_order
    trace = _Trace()
    trace.emit(
        StepKind.INITIAL, "setup", None, (),
        "Initial system: one equation per state summing its incoming transitions "
        f"(q·x for each move q --x--> state), plus ε for the start state {system.start}. "
        f"Elimination order: {', '.join(order)}.",
        None, system, system,
    )

    # -- forward elimination ---------------------------------------------
    current = system
    for i, v in enumerate(order):
        current = _arden_step(current, v, trace, "elimination")
        v_eq = current.get(v)
        later = order[i + 1:]
        targets = [w for w in later if v in current.get(w).variables()]
        if not targets:
            continue
        before = current
        new_texts = []
        for w in targets:
            new_eq = _check_size(current.get(w).substitute(v, v_eq))
            current = current.replace(new_eq)
            new_texts.append(new_eq.to_text(state_order))
        trace.emit(
            StepKind.SUBSTITUTE, "elimination", v, targets,
            f"Substitute {v_eq.to_text(state_order)} into the equation(s) of "
            f"{', '.join(targets)}, which become: " + "; ".join(new_texts) + ".",
            f"{state_to_latex(v)} = {v_eq.rhs_latex(state_order)} \\;\\Rightarrow\\; "
            + ",\\; ".join(
                f"{state_to_latex(w)} = {current.get(w).rhs_latex(state_order)}" for w in targets
            ),
            before, current,
        )

    if not current.get(order[-1]).is_closed():  # pragma: no cover - invariant
        raise SolverError(f"internal error: {order[-1]} is not closed after elimination")

    # -- back-substitution -----------------------------------------------
    for i in range(len(order) - 2, -1, -1):
        v = order[i]
        before = current
        eq = current.get(v)
        used = [w for w in order[i + 1:] if w in eq.variables()]
        for w in used:
            eq = _check_size(eq.substitute(w, current.get(w)))
        current = current.replace(eq)
        if eq.has_self_reference():  # pragma: no cover - impossible after elimination
            current = _arden_step(current, v, trace, "back_substitution")
            eq = current.get(v)
        if not eq.is_closed():  # pragma: no cover - invariant
            raise SolverError(f"internal error: {v} is not closed after back-substitution")
        if used:
            explanation = (
                f"Back-substitute the closed form(s) of {', '.join(used)} into the equation "
                f"of {v}: {eq.to_text(state_order)}."
            )
        else:
            explanation = f"{v} is already closed: {eq.to_text(state_order)}."
        trace.emit(
            StepKind.BACK_SUBSTITUTE, "back_substitution", v, (v,) if used else (),
            explanation, eq.to_latex(state_order), before, current,
        )

    closed = {s: current.get(s).const for s in state_order}

    # -- combine finals ------------------------------------------------------
    finals = [s for s in state_order if s in system.finals]
    regex = simplify(union(*(closed[s] for s in finals)))
    if finals:
        parts = " + ".join(f"({to_text(closed[s])})" for s in finals)
        explanation = (
            f"The language is the union of the closed forms of the final state(s) "
            f"{', '.join(finals)}: {parts}, which simplifies to {to_text(regex)}."
        )
        rule = (
            " + ".join(state_to_latex(s) for s in finals) + f" = {to_latex(regex)}"
        )
    else:
        explanation = (
            "The automaton has no final states, so it accepts no string: "
            "the regular expression is ∅ (the empty language)."
        )
        rule = r"\varnothing"
    trace.emit(
        StepKind.COMBINE_FINALS, "result", None, tuple(finals),
        explanation, rule, current, current,
    )
    trace.emit(
        StepKind.RESULT, "result", None, (),
        f"Final regular expression: {to_text(regex)}.",
        f"r = {to_latex(regex)}", current, current,
    )
    return SolveResult(regex, ReadOnlyMap(closed), order, tuple(trace.steps))


def solve_all_orders(system: EquationSystem, limit: int = 24) -> list[SolveResult]:
    """Solve the system with every elimination order (default order first).

    Only for small systems (at most MAX_STATES_ALL_ORDERS states); returns at
    most ``limit`` results. All results describe the same language.

    Raises:
        SolverError: if the system has more than MAX_STATES_ALL_ORDERS states.
    """
    n = len(system.state_order)
    if n > MAX_STATES_ALL_ORDERS:
        raise SolverError(
            f"solve_all_orders supports at most {MAX_STATES_ALL_ORDERS} states "
            f"(this one has {n}, i.e. {math.factorial(n)} possible orders)"
        )
    first = default_order(system)
    orders = [first] + [p for p in itertools.permutations(system.state_order) if p != first]
    return [solve(system, o) for o in orders[: max(limit, 0)]]
