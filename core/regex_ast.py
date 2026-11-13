"""Regular expression abstract syntax tree.

Responsibilities: the regex AST (Empty, Epsilon, Symbol, Union, Concat, Star),
smart constructors, the simplifier, ``nullable()``, and printers (text, LaTeX,
Python ``re``).

Other modules must build regexes **only** through the smart constructors
:func:`union`, :func:`concat`, :func:`star`, :func:`sym` and the constants
:data:`EPS` and :data:`EMPTY`. The constructors keep every expression in a
light canonical form (no ∅ inside unions, no ε inside concatenations, no
nested unions/concatenations, no duplicate union members); :func:`simplify`
applies deeper algebraic identities on top of that.

Printing order is stable: union members are sorted with ``ε`` first and the
remaining members ordered by their text rendering, so ``a + ε`` always prints
as ``ε+a`` and ``b + a`` always prints as ``a+b``.
"""

from __future__ import annotations

import re
from abc import ABC
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

from core import InvalidAutomatonError

__all__ = [
    "Regex",
    "Empty",
    "Epsilon",
    "Symbol",
    "Union",
    "Concat",
    "Star",
    "EMPTY",
    "EPS",
    "RESERVED_CHARS",
    "sym",
    "union",
    "concat",
    "star",
    "simplify",
    "nullable",
    "is_empty_language",
    "alphabet",
    "size",
    "sort_key",
    "to_text",
    "to_latex",
    "to_python_re",
]

#: Characters that may never be used as alphabet symbols (whitespace is also rejected).
RESERVED_CHARS: frozenset[str] = frozenset("+*()|ε∅")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Regex(ABC):
    """Abstract base class of every regular-expression AST node.

    Concrete nodes are frozen (immutable, hashable) dataclasses.
    """

    __slots__ = ()

    def __str__(self) -> str:
        """Render the expression in the project's text notation (see :func:`to_text`)."""
        return to_text(self)


@dataclass(frozen=True, slots=True)
class Empty(Regex):
    """The empty language ∅ (matches no string at all)."""


@dataclass(frozen=True, slots=True)
class Epsilon(Regex):
    """The language {ε} containing only the empty string."""


@dataclass(frozen=True, slots=True)
class Symbol(Regex):
    """A single alphabet character.

    Raises:
        InvalidAutomatonError: if ``char`` is not exactly one character, is
            whitespace, or is one of the reserved characters ``+ * ( ) | ε ∅``.
    """

    char: str

    def __post_init__(self) -> None:
        """Validate the alphabet-symbol convention from CLAUDE.md."""
        c = self.char
        if not isinstance(c, str) or len(c) != 1:
            raise InvalidAutomatonError(
                f"alphabet symbols must be single characters, got {c!r}"
            )
        if c.isspace() or c in RESERVED_CHARS:
            raise InvalidAutomatonError(f"reserved character {c!r} cannot be a symbol")


@dataclass(frozen=True, slots=True)
class Union(Regex):
    """N-ary, unordered union ``R1 + R2 + ...`` (at least 2 members, no nested Union).

    Members are stored in a frozenset, so duplicates collapse and equality is
    independent of order.
    """

    items: frozenset[Regex]

    def __post_init__(self) -> None:
        """Check the structural invariants (use :func:`union` to build safely)."""
        if not isinstance(self.items, frozenset):
            raise TypeError("Union.items must be a frozenset")
        if len(self.items) < 2:
            raise ValueError("Union needs at least 2 members; use union()")
        for item in self.items:
            if not isinstance(item, Regex):
                raise TypeError(f"Union member is not a Regex: {item!r}")
            if isinstance(item, Union):
                raise ValueError("Union must be flat; use union()")


@dataclass(frozen=True, slots=True)
class Concat(Regex):
    """N-ary, ordered concatenation ``R1 R2 ...`` (at least 2 parts, no nested Concat)."""

    items: tuple[Regex, ...]

    def __post_init__(self) -> None:
        """Check the structural invariants (use :func:`concat` to build safely)."""
        if not isinstance(self.items, tuple):
            raise TypeError("Concat.items must be a tuple")
        if len(self.items) < 2:
            raise ValueError("Concat needs at least 2 parts; use concat()")
        for item in self.items:
            if not isinstance(item, Regex):
                raise TypeError(f"Concat part is not a Regex: {item!r}")
            if isinstance(item, Concat):
                raise ValueError("Concat must be flat; use concat()")


@dataclass(frozen=True, slots=True)
class Star(Regex):
    """Kleene star ``R*``."""

    inner: Regex

    def __post_init__(self) -> None:
        """Check that the operand is a Regex."""
        if not isinstance(self.inner, Regex):
            raise TypeError(f"Star operand is not a Regex: {self.inner!r}")


#: The empty-language constant ∅.
EMPTY: Regex = Empty()
#: The empty-string constant ε.
EPS: Regex = Epsilon()


# ---------------------------------------------------------------------------
# Smart constructors
# ---------------------------------------------------------------------------


def sym(c: str) -> Regex:
    """Build the single-symbol regex ``c`` (validated against the alphabet rules)."""
    return Symbol(c)


def union(*rs: Regex) -> Regex:
    """Build ``r1 + r2 + ...`` in canonical form.

    Rules (each is a language identity):
      * ``∅ + R = R``          — ∅ members are dropped;
      * ``(R + S) + T = R + S + T`` — nested unions are flattened;
      * ``R + R = R``          — duplicates collapse (frozenset);
      * no members ⇒ ``∅``; one member ⇒ that member.
    """
    members: set[Regex] = set()
    for r in rs:
        if isinstance(r, Empty):
            continue
        if isinstance(r, Union):
            members.update(r.items)
        else:
            members.add(r)
    if not members:
        return EMPTY
    if len(members) == 1:
        return next(iter(members))
    return Union(frozenset(members))


def concat(*rs: Regex) -> Regex:
    """Build ``r1 r2 ...`` in canonical form.

    Rules (each is a language identity):
      * ``R ∅ = ∅ R = ∅``      — any ∅ part annihilates;
      * ``R ε = ε R = R``      — ε parts are dropped;
      * ``(R S) T = R S T``    — nested concatenations are flattened;
      * no parts ⇒ ``ε``; one part ⇒ that part.
    """
    parts: list[Regex] = []
    for r in rs:
        if isinstance(r, Empty):
            return EMPTY
        if isinstance(r, Epsilon):
            continue
        if isinstance(r, Concat):
            parts.extend(r.items)
        else:
            parts.append(r)
    if not parts:
        return EPS
    if len(parts) == 1:
        return parts[0]
    return Concat(tuple(parts))


def star(r: Regex) -> Regex:
    """Build ``r*`` in canonical form.

    Rules: ``∅* = ε``, ``ε* = ε``, ``(R*)* = R*``.
    """
    if isinstance(r, (Empty, Epsilon)):
        return EPS
    if isinstance(r, Star):
        return r
    return Star(r)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def nullable(r: Regex) -> bool:
    """Return True iff ε ∈ L(r).

    ∅ → no, ε → yes, symbol → no, union → any member, concat → all parts,
    star → yes. This is the test used for Arden's precondition
    (R = Q + RP has the unique solution QP* only when P is not nullable).
    """
    if isinstance(r, (Epsilon, Star)):
        return True
    if isinstance(r, (Empty, Symbol)):
        return False
    if isinstance(r, Union):
        return any(nullable(i) for i in r.items)
    if isinstance(r, Concat):
        return all(nullable(i) for i in r.items)
    raise TypeError(f"not a Regex: {r!r}")


def is_empty_language(r: Regex) -> bool:
    """Return True iff L(r) = ∅.

    ∅ → yes, ε / symbol / star → no, union → all members, concat → any part.
    """
    if isinstance(r, Empty):
        return True
    if isinstance(r, (Epsilon, Symbol, Star)):
        return False
    if isinstance(r, Union):
        return all(is_empty_language(i) for i in r.items)
    if isinstance(r, Concat):
        return any(is_empty_language(i) for i in r.items)
    raise TypeError(f"not a Regex: {r!r}")


def alphabet(r: Regex) -> frozenset[str]:
    """Return the set of symbol characters occurring syntactically in ``r``."""
    if isinstance(r, Symbol):
        return frozenset(r.char)
    if isinstance(r, (Empty, Epsilon)):
        return frozenset()
    if isinstance(r, Star):
        return alphabet(r.inner)
    if isinstance(r, (Union, Concat)):
        return frozenset().union(*(alphabet(i) for i in r.items))
    raise TypeError(f"not a Regex: {r!r}")


def size(r: Regex) -> int:
    """Return the number of AST nodes in ``r`` (the measure simplify never increases)."""
    if isinstance(r, (Empty, Epsilon, Symbol)):
        return 1
    if isinstance(r, Star):
        return 1 + size(r.inner)
    if isinstance(r, (Union, Concat)):
        return 1 + sum(size(i) for i in r.items)
    raise TypeError(f"not a Regex: {r!r}")


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------


def sort_key(r: Regex) -> tuple[int, str, str]:
    """Deterministic ordering key for union members: ε first, then by text rendering."""
    return (0 if isinstance(r, Epsilon) else 1, to_text(r), repr(r))


def _sorted_members(u: Union) -> list[Regex]:
    """Union members in stable printing order."""
    return sorted(u.items, key=sort_key)


def _is_atomic(r: Regex) -> bool:
    """True for nodes that never need parentheses under a star."""
    return isinstance(r, (Empty, Epsilon, Symbol))


@lru_cache(maxsize=8192)
def to_text(r: Regex) -> str:
    """Render ``r`` in text notation: ``+`` union, juxtaposition, ``*``, ``ε``, ``∅``.

    Uses minimal parentheses for the precedence star > concat > union: a union
    inside a concatenation is parenthesised, and the operand of a star is
    parenthesised unless it is atomic, e.g. ``a(a+b)*b``, ``(ab)*``, ``a*b*``.
    """
    if isinstance(r, Empty):
        return "∅"
    if isinstance(r, Epsilon):
        return "ε"
    if isinstance(r, Symbol):
        return r.char
    if isinstance(r, Union):
        return "+".join(to_text(m) for m in _sorted_members(r))
    if isinstance(r, Concat):
        return "".join(
            f"({to_text(p)})" if isinstance(p, Union) else to_text(p) for p in r.items
        )
    if isinstance(r, Star):
        inner = to_text(r.inner)
        return f"{inner}*" if _is_atomic(r.inner) else f"({inner})*"
    raise TypeError(f"not a Regex: {r!r}")


_LATEX_ESCAPES: dict[str, str] = {
    "\\": r"\backslash ",
    "{": r"\{",
    "}": r"\}",
    "#": r"\#",
    "$": r"\$",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "^": r"\hat{}",
    "~": r"\sim ",
}


def _join_latex(parts: Iterable[str]) -> str:
    """Juxtapose LaTeX fragments, inserting a space after a bare control word."""
    out = ""
    for p in parts:
        if out and re.search(r"\\[A-Za-z]+$", out) and p[:1].isalpha():
            out += " "
        out += p
    return out


def to_latex(r: Regex) -> str:
    """Render ``r`` as LaTeX math (same structure and parentheses as :func:`to_text`).

    ``∅`` → ``\\varnothing``, ``ε`` → ``\\varepsilon``, star → ``^{*}``,
    union → ``+``. The few LaTeX-special ASCII characters are escaped.
    """
    if isinstance(r, Empty):
        return r"\varnothing"
    if isinstance(r, Epsilon):
        return r"\varepsilon"
    if isinstance(r, Symbol):
        return _LATEX_ESCAPES.get(r.char, r.char)
    if isinstance(r, Union):
        return "+".join(to_latex(m) for m in _sorted_members(r))
    if isinstance(r, Concat):
        return _join_latex(
            f"({to_latex(p)})" if isinstance(p, Union) else to_latex(p) for p in r.items
        )
    if isinstance(r, Star):
        inner = to_latex(r.inner)
        base = inner if _is_atomic(r.inner) else f"({inner})"
        return f"{base}^{{*}}"
    raise TypeError(f"not a Regex: {r!r}")


def to_python_re(r: Regex) -> str:
    """Render ``r`` as a Python ``re`` pattern intended for ``re.fullmatch``.

    union → ``(?:A|B)``, symbols ``re.escape``'d, ``ε`` → ``(?:)``,
    ``∅`` → ``(?!)`` (never matches), star → ``(?:...)*``.
    """
    if isinstance(r, Empty):
        return "(?!)"
    if isinstance(r, Epsilon):
        return "(?:)"
    if isinstance(r, Symbol):
        return re.escape(r.char)
    if isinstance(r, Union):
        return "(?:" + "|".join(to_python_re(m) for m in _sorted_members(r)) + ")"
    if isinstance(r, Concat):
        return "".join(to_python_re(p) for p in r.items)
    if isinstance(r, Star):
        return f"(?:{to_python_re(r.inner)})*"
    raise TypeError(f"not a Regex: {r!r}")


# ---------------------------------------------------------------------------
# Simplifier
# ---------------------------------------------------------------------------
#
# Every rewrite below is a language identity AND strictly decreases size(),
# which guarantees termination, size(simplify(r)) <= size(r), and (because the
# result is a fixpoint of a full pass) idempotence.


def _parts(r: Regex) -> tuple[Regex, ...]:
    """View ``r`` as a sequence of concatenated factors."""
    return r.items if isinstance(r, Concat) else (r,)


def _rewrite_star(s: Regex) -> Regex | None:
    """(ε + R)* = R*  — ε inside a starred union is redundant."""
    if isinstance(s, Star) and isinstance(s.inner, Union) and EPS in s.inner.items:
        return star(union(*(m for m in s.inner.items if m != EPS)))
    return None


def _rewrite_concat(c: Regex) -> Regex | None:
    """R*·R* = R*  — merge adjacent identical starred factors."""
    if not isinstance(c, Concat):
        return None
    out: list[Regex] = []
    for p in c.items:
        if out and isinstance(p, Star) and out[-1] == p:
            continue
        out.append(p)
    if len(out) < len(c.items):
        return concat(*out)
    return None


def _star_absorb(m: Regex) -> Regex | None:
    """If ``m`` is ``R·R*`` or ``R*·R``, return ``R*`` (else None)."""
    if not isinstance(m, Concat):
        return None
    first, last = m.items[0], m.items[-1]
    if isinstance(last, Star) and concat(*m.items[:-1]) == last.inner:
        return last
    if isinstance(first, Star) and concat(*m.items[1:]) == first.inner:
        return first
    return None


def _common_prefix(seqs: list[tuple[Regex, ...]]) -> int:
    """Length of the longest common prefix of the given factor sequences."""
    n = min(len(s) for s in seqs)
    k = 0
    while k < n and all(s[k] == seqs[0][k] for s in seqs):
        k += 1
    return k


def _factor_candidates(u: Union) -> list[Regex]:
    """All left/right common-factor extractions of ``u`` (not yet size-filtered).

    Left:  a·X + a·Y = a·(X + Y)   (concatenation distributes over union)
    Right: X·a + Y·a = (X + Y)·a
    The longest common prefix/suffix of each group is factored out at once.
    """
    members = sorted(u.items, key=sort_key)
    candidates: list[Regex] = []
    for side in ("left", "right"):
        groups: dict[Regex, list[Regex]] = {}
        for m in members:
            ps = _parts(m)
            key = ps[0] if side == "left" else ps[-1]
            groups.setdefault(key, []).append(m)
        for key in sorted(groups, key=sort_key):
            group = groups[key]
            if len(group) < 2:
                continue
            others = [m for m in members if m not in group]
            if side == "left":
                seqs = [_parts(m) for m in group]
                k = _common_prefix(seqs)
                common = seqs[0][:k]
                rest = union(*(concat(*s[k:]) for s in seqs))
                factored = concat(*common, rest)
            else:
                seqs = [tuple(reversed(_parts(m))) for m in group]
                k = _common_prefix(seqs)
                common = tuple(reversed(seqs[0][:k]))
                rest = union(*(concat(*reversed(s[k:])) for s in seqs))
                factored = concat(rest, *common)
            candidates.append(union(factored, *others))
    return candidates


def _rewrite_union(u: Regex) -> Regex | None:
    """One rewrite step on a union node (first applicable rule wins).

    1. ε + R = R when R is nullable (covers ε + R* = R*).
    2. ε + R·R* = R*  and  ε + R*·R = R*.
    3. Common left / right factor extraction, only if it strictly reduces size().
    """
    if not isinstance(u, Union):
        return None
    items = u.items
    if EPS in items:
        rest = [m for m in items if m != EPS]
        # Rule 1: ε is redundant next to a nullable member.
        if any(nullable(m) for m in rest):
            return union(*rest)
        # Rule 2: ε + R·R* = ε + R*·R = R*.
        for m in sorted(rest, key=sort_key):
            absorbed = _star_absorb(m)
            if absorbed is not None:
                return union(absorbed, *(x for x in rest if x != m))
    # Rule 3: factoring, accepted only when it strictly shrinks the expression.
    current = size(u)
    best: Regex | None = None
    for cand in _factor_candidates(u):
        if size(cand) < current and (
            best is None or (size(cand), sort_key(cand)) < (size(best), sort_key(best))
        ):
            best = cand
    return best


_LOCAL_RULES: tuple[Callable[[Regex], Regex | None], ...] = (
    _rewrite_star,
    _rewrite_concat,
    _rewrite_union,
)


def _rewrite_local(r: Regex) -> Regex:
    """Apply node-level rewrites at the root of ``r`` until none applies."""
    while True:
        for rule in _LOCAL_RULES:
            nxt = rule(r)
            if nxt is not None:
                r = nxt
                break
        else:
            return r


def _pass(r: Regex) -> Regex:
    """One bottom-up simplification pass: rebuild children, then rewrite the node."""
    if isinstance(r, (Empty, Epsilon, Symbol)):
        return r
    if isinstance(r, Star):
        return _rewrite_local(star(_pass(r.inner)))
    if isinstance(r, Union):
        return _rewrite_local(union(*(_pass(i) for i in r.items)))
    if isinstance(r, Concat):
        return _rewrite_local(concat(*(_pass(i) for i in r.items)))
    raise TypeError(f"not a Regex: {r!r}")


def simplify(r: Regex) -> Regex:
    """Simplify ``r`` bottom-up to a fixpoint using language-preserving identities.

    Identities used (beyond those built into the smart constructors):
      * ε + R·R* = R*,  ε + R*·R = R*
      * ε + R = R when R is nullable (in particular ε + R* = R*)
      * (ε + R)* = R*
      * R*·R* = R*
      * (R*)* = R*
      * a·X + a·Y = a·(X + Y) and X·a + Y·a = (X + Y)·a, only if size() shrinks

    The result is language-equivalent to ``r``, never larger (by :func:`size`),
    and ``simplify(simplify(r)) == simplify(r)``.
    """
    current = r
    while True:
        nxt = _pass(current)
        if nxt == current:
            return current
        if size(nxt) >= size(current):  # pragma: no cover - defensive guard
            raise AssertionError(
                f"simplify made no size progress: {to_text(current)} -> {to_text(nxt)}"
            )
        current = nxt
