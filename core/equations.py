"""Left-linear equation systems derived from an automaton.

Responsibilities: Term / Equation / EquationSystem, the builder from an
automaton, substitution, like-term collection, and printers.

Every equation is kept in **left-linear form**::

    q = q_1·C_1 + q_2·C_2 + ... + K

where each ``C_k`` is a constant regex coefficient and ``K`` is the constant
term (∅ when absent). Like terms are always collected: two terms with the same
variable merge into one whose coefficient is the union of both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from core import ArdenPreconditionError, SolverError
from core.automaton import Automaton
from core.regex_ast import (
    EMPTY,
    EPS,
    Empty,
    Epsilon,
    Regex,
    Union,
    concat,
    simplify,
    sort_key,
    sym,
    to_latex,
    to_text,
    union,
)

__all__ = [
    "Term",
    "Equation",
    "EquationSystem",
    "build_equations",
    "state_to_latex",
]

#: A right-hand-side term ``(variable, coefficient)`` meaning ``variable·coefficient``.
Term = tuple[str, Regex]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _natural_key(name: str) -> tuple[object, ...]:
    """Sort key ordering ``q2`` before ``q10`` (digit runs compare numerically)."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", name)
        if part
    )


def _collect(terms: Iterable[Term]) -> tuple[Term, ...]:
    """Collect like terms: q·C1 + q·C2 = q·(C1 + C2); drop ∅ coefficients.

    Returns the terms in a canonical (natural variable-name) order so that
    equality and hashing do not depend on insertion order.
    """
    merged: dict[str, Regex] = {}
    for var, coeff in terms:
        merged[var] = union(merged[var], coeff) if var in merged else coeff
    return tuple(
        (var, merged[var])
        for var in sorted(merged, key=_natural_key)
        if not isinstance(merged[var], Empty)
    )


def _order_terms(
    terms: Sequence[Term], state_order: Sequence[str] | None
) -> list[Term]:
    """Terms in automaton state order; unknown variables follow in natural order."""
    if not state_order:
        return list(terms)
    rank = {s: i for i, s in enumerate(state_order)}
    return sorted(terms, key=lambda t: (rank.get(t[0], len(rank)), _natural_key(t[0])))


def _const_parts(const: Regex) -> list[Regex]:
    """Split a constant into the summands printed after the variable terms."""
    if isinstance(const, Empty):
        return []
    if isinstance(const, Union):
        return sorted(const.items, key=sort_key)
    return [const]


_LATEX_SPECIAL = re.compile(r"([\\{}#$%&_^~])")


def state_to_latex(name: str) -> str:
    r"""Render a state name in LaTeX.

    ``q12`` → ``q_{12}`` (one letter followed by digits); anything else, such
    as ``A`` or ``start``, → ``\mathit{start}`` (special characters escaped).
    """
    m = re.fullmatch(r"([A-Za-z])(\d+)", name)
    if m:
        return f"{m.group(1)}_{{{m.group(2)}}}"
    return r"\mathit{" + _LATEX_SPECIAL.sub(r"\\\1", name) + "}"


def _coeff_text(c: Regex) -> str:
    """Coefficient text; a union is parenthesised, ε coefficient prints as nothing."""
    if isinstance(c, Epsilon):
        return ""
    return f"({to_text(c)})" if isinstance(c, Union) else to_text(c)


def _coeff_latex(c: Regex) -> str:
    """Coefficient LaTeX; a union is parenthesised, ε coefficient prints as nothing."""
    if isinstance(c, Epsilon):
        return ""
    return f"({to_latex(c)})" if isinstance(c, Union) else to_latex(c)


# ---------------------------------------------------------------------------
# Equation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Equation:
    """A left-linear equation ``lhs = Σ var·coeff + const``.

    ``coeffs`` may be given as a mapping or any iterable of ``(var, coeff)``
    pairs; it is stored as a canonical tuple of pairs with like terms
    collected and ∅ coefficients dropped, so equations are hashable and
    equality does not depend on term order.
    """

    lhs: str
    coeffs: tuple[Term, ...] = ()
    const: Regex = EMPTY

    def __post_init__(self) -> None:
        """Normalize ``coeffs``: collect like terms, drop ∅, canonical order."""
        raw = self.coeffs.items() if isinstance(self.coeffs, Mapping) else self.coeffs
        object.__setattr__(self, "coeffs", _collect(raw))

    # -- queries --------------------------------------------------------------

    @property
    def coeff_map(self) -> dict[str, Regex]:
        """A fresh ``{variable: coefficient}`` dict (mutating it does not affect self)."""
        return dict(self.coeffs)

    def coeff(self, var: str) -> Regex:
        """Coefficient of ``var`` on the right-hand side (∅ if ``var`` is absent)."""
        for v, c in self.coeffs:
            if v == var:
                return c
        return EMPTY

    def variables(self) -> frozenset[str]:
        """Variables appearing on the right-hand side."""
        return frozenset(v for v, _ in self.coeffs)

    def self_coeff(self) -> Regex:
        """Coefficient P of ``lhs`` in ``lhs = Q + lhs·P`` (∅ if there is no self-term)."""
        return self.coeff(self.lhs)

    def has_self_reference(self) -> bool:
        """True iff ``lhs`` occurs on its own right-hand side (Arden is then needed)."""
        return self.lhs in self.variables()

    def is_closed(self) -> bool:
        """True iff the right-hand side contains no variables (a closed-form regex)."""
        return not self.coeffs

    # -- transformations ---------------------------------------------------

    def substitute(self, var: str, rhs: "Equation") -> "Equation":
        """Replace ``var`` by the right-hand side of ``rhs`` (left-linear substitution).

        With ``rhs: var = Σ_k q_k·C_k + K``, each term ``var·C`` becomes
        ``Σ_k q_k·(C_k·C) + K·C``. This is just distributivity of concatenation
        over union, so the left-linear form is preserved. Like terms are
        collected and every coefficient (and the constant) is simplified.

        Raises:
            ValueError: if ``rhs`` is not the equation of ``var``.
        """
        if rhs.lhs != var:
            raise ValueError(
                f"substitute({var!r}, ...) needs the equation of {var!r}, got one for {rhs.lhs!r}"
            )
        c = self.coeff(var)
        if isinstance(c, Empty):
            return self
        terms: list[Term] = [(v, k) for v, k in self.coeffs if v != var]
        terms.extend((qk, concat(ck, c)) for qk, ck in rhs.coeffs)
        const = union(self.const, concat(rhs.const, c))
        return Equation(self.lhs, tuple(terms), const).simplified()

    def simplified(self) -> "Equation":
        """Copy with every coefficient and the constant passed through ``simplify``."""
        return Equation(
            self.lhs,
            tuple((v, simplify(c)) for v, c in self.coeffs),
            simplify(self.const),
        )

    # -- printers -------------------------------------------------------------

    def rhs_text(self, state_order: Sequence[str] | None = None) -> str:
        """Right-hand side text, e.g. ``q0·0 + q1·1 + ε`` (``∅`` if empty)."""
        parts = [
            f"{v}·{_coeff_text(c)}" if _coeff_text(c) else v
            for v, c in _order_terms(self.coeffs, state_order)
        ]
        parts.extend(to_text(k) for k in _const_parts(self.const))
        return " + ".join(parts) if parts else "∅"

    def rhs_latex(self, state_order: Sequence[str] | None = None) -> str:
        """Right-hand side LaTeX, e.g. ``q_{0}0 + q_{1}1 + \\varepsilon``."""
        parts = [
            state_to_latex(v) + _coeff_latex(c)
            for v, c in _order_terms(self.coeffs, state_order)
        ]
        parts.extend(to_latex(k) for k in _const_parts(self.const))
        return " + ".join(parts) if parts else r"\varnothing"

    def to_text(self, state_order: Sequence[str] | None = None) -> str:
        """Text form, e.g. ``q1 = q0·0 + q1·1 + ε``: variables first (in
        ``state_order`` if given, else natural name order), constant last."""
        return f"{self.lhs} = {self.rhs_text(state_order)}"

    def to_latex(self, state_order: Sequence[str] | None = None) -> str:
        """LaTeX form, e.g. ``q_{1} = q_{0}0 + q_{1}1 + \\varepsilon`` (no dot)."""
        return f"{state_to_latex(self.lhs)} = {self.rhs_latex(state_order)}"

    def __str__(self) -> str:
        """Delegates to :meth:`to_text`."""
        return self.to_text()


# ---------------------------------------------------------------------------
# EquationSystem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquationSystem:
    """One equation per state, in automaton state order."""

    equations: tuple[Equation, ...]
    state_order: tuple[str, ...]
    start: str
    finals: frozenset[str]

    def __post_init__(self) -> None:
        """Coerce containers and check that equations match ``state_order`` one-to-one."""
        object.__setattr__(self, "equations", tuple(self.equations))
        object.__setattr__(self, "state_order", tuple(self.state_order))
        object.__setattr__(self, "finals", frozenset(self.finals))
        lhs = tuple(e.lhs for e in self.equations)
        if lhs != self.state_order:
            raise ValueError(
                f"equations {lhs} do not match the state order {self.state_order}"
            )

    def get(self, state: str) -> Equation:
        """The equation whose left-hand side is ``state``.

        Raises:
            SolverError: if ``state`` has no equation in this system.
        """
        for e in self.equations:
            if e.lhs == state:
                return e
        raise SolverError(f"no equation for state {state!r}")

    def replace(self, eq: Equation) -> "EquationSystem":
        """New system in which the equation for ``eq.lhs`` is replaced by ``eq``."""
        self.get(eq.lhs)  # raises if unknown
        return EquationSystem(
            tuple(eq if e.lhs == eq.lhs else e for e in self.equations),
            self.state_order,
            self.start,
            self.finals,
        )

    def to_text(self) -> str:
        """One equation per line, terms in state order."""
        return "\n".join(e.to_text(self.state_order) for e in self.equations)

    def to_latex(self) -> str:
        """An ``aligned`` environment, one equation per line, aligned on ``&=``."""
        lines = [
            f"{state_to_latex(e.lhs)} &= {e.rhs_latex(self.state_order)}"
            for e in self.equations
        ]
        return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}"

    def __str__(self) -> str:
        """Delegates to :meth:`to_text`."""
        return self.to_text()


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_equations(a: Automaton) -> EquationSystem:
    """Build the left-linear equation system of an ε-free automaton.

    Theory (CLAUDE.md, "Applying it to an FA", step 1): for every state ``qi``
    the right-hand side sums the incoming transitions — each ``qj --x--> qi``
    contributes the term ``qj·x`` — and the start state's equation also gets
    the constant ``ε``. A state with no incoming transitions that is not the
    start gets ``q = ∅``. Parallel transitions collect into one coefficient,
    e.g. ``q0 --a--> q1`` and ``q0 --b--> q1`` give ``q1 = q0·(a+b)``.

    Raises:
        ArdenPreconditionError: if ``a`` has ε-transitions (run ``normalize()`` first).
    """
    if a.has_epsilon():
        raise ArdenPreconditionError(
            "the automaton has ε-transitions; run normalize() first"
        )
    incoming: dict[str, list[Term]] = {s: [] for s in a.states}
    for src, x, dst in a.transitions:
        incoming[dst].append((src, sym(x)))
    equations = tuple(
        Equation(s, tuple(incoming[s]), EPS if s == a.start else EMPTY)
        for s in a.states
    )
    return EquationSystem(equations, a.states, a.start, a.finals)
