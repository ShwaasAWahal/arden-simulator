"""Tests for core.regex_ast: constructors, printers, queries, simplifier."""

from __future__ import annotations

import itertools
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core import InvalidAutomatonError
from core.regex_ast import (
    EMPTY,
    EPS,
    Concat,
    Regex,
    Star,
    Symbol,
    Union,
    alphabet,
    concat,
    is_empty_language,
    nullable,
    simplify,
    size,
    star,
    sym,
    to_latex,
    to_python_re,
    to_text,
    union,
)

a, b, c, d = sym("a"), sym("b"), sym("c"), sym("d")


def all_strings(alpha: str, max_len: int) -> list[str]:
    """Every string over ``alpha`` of length 0..max_len."""
    return [
        "".join(t)
        for n in range(max_len + 1)
        for t in itertools.product(alpha, repeat=n)
    ]


STRINGS_AB_6 = all_strings("ab", 6)


def matches(r: Regex, s: str) -> bool:
    """Does ``r`` (via its Python re rendering) fully match ``s``?"""
    return re.fullmatch(to_python_re(r), s) is not None


# ---------------------------------------------------------------------------
# 1. Constructor rules
# ---------------------------------------------------------------------------


class TestUnionConstructor:
    def test_drops_empty(self) -> None:
        assert union(a, EMPTY) == a

    def test_flattens_nested(self) -> None:
        r = union(a, union(b, c))
        assert isinstance(r, Union)
        assert r.items == frozenset({a, b, c})
        assert not any(isinstance(i, Union) for i in r.items)

    def test_deduplicates(self) -> None:
        assert union(a, a) == a

    def test_zero_members_is_empty(self) -> None:
        assert union() == EMPTY
        assert union(EMPTY, EMPTY) == EMPTY

    def test_one_member_is_that_member(self) -> None:
        assert union(a) == a


class TestConcatConstructor:
    def test_empty_annihilates(self) -> None:
        assert concat(a, EMPTY) == EMPTY
        assert concat(EMPTY, a, b) == EMPTY

    def test_drops_epsilon(self) -> None:
        assert concat(a, EPS, b) == Concat((a, b))

    def test_flattens_nested(self) -> None:
        assert concat(a, concat(b, c)) == Concat((a, b, c))
        assert concat(concat(a, b), c) == Concat((a, b, c))

    def test_zero_parts_is_epsilon(self) -> None:
        assert concat() == EPS
        assert concat(EPS, EPS) == EPS

    def test_one_part_is_that_part(self) -> None:
        assert concat(a) == a
        assert concat(EPS, a) == a


class TestStarConstructor:
    def test_star_empty_is_epsilon(self) -> None:
        assert star(EMPTY) == EPS

    def test_star_epsilon_is_epsilon(self) -> None:
        assert star(EPS) == EPS

    def test_star_star_collapses(self) -> None:
        assert star(star(a)) == star(a)


class TestSymbol:
    @pytest.mark.parametrize("bad", ["", "ab", "+", "*", "(", ")", "|", "ε", "∅", " ", "\t"])
    def test_rejects_invalid_symbols(self, bad: str) -> None:
        with pytest.raises(InvalidAutomatonError):
            sym(bad)

    def test_structural_invariants(self) -> None:
        with pytest.raises(ValueError):
            Union(frozenset({a}))
        with pytest.raises(ValueError):
            Concat((a,))
        with pytest.raises(ValueError):
            Concat((a, Concat((b, c))))


def test_nodes_are_frozen_and_hashable() -> None:
    r = concat(a, star(union(a, b)))
    assert hash(r) == hash(concat(a, star(union(b, a))))
    with pytest.raises(AttributeError):
        r.items = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Union equality is order-independent
# ---------------------------------------------------------------------------


def test_union_order_independent() -> None:
    assert union(a, b) == union(b, a)
    assert union(a, union(b, c)) == union(union(c, a), b)
    assert concat(a, b) != concat(b, a)  # ...but concatenation is ordered


# ---------------------------------------------------------------------------
# 3. Printers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("r", "text"),
    [
        (concat(a, star(union(a, b)), b), "a(a+b)*b"),
        (star(concat(a, b)), "(ab)*"),
        (concat(star(a), star(b)), "a*b*"),
        (EPS, "ε"),
        (EMPTY, "∅"),
        # Documented stable order: ε first, then members sorted by their text.
        (union(a, EPS), "ε+a"),
        (union(b, a), "a+b"),
        (union(concat(a, b), b, EPS), "ε+ab+b"),
        (concat(union(a, b), union(a, b)), "(a+b)(a+b)"),
        (star(star(a)), "a*"),
        (Star(Star(a)), "(a*)*"),
        (union(star(a), concat(a, b)), "a*+ab"),
    ],
)
def test_to_text(r: Regex, text: str) -> None:
    assert to_text(r) == text
    assert str(r) == text


@pytest.mark.parametrize(
    ("r", "latex"),
    [
        (concat(a, star(union(a, b)), b), "a(a+b)^{*}b"),
        (star(concat(a, b)), "(ab)^{*}"),
        (union(a, EPS), r"\varepsilon+a"),
        (EMPTY, r"\varnothing"),
        (sym("_"), r"\_"),
    ],
)
def test_to_latex(r: Regex, latex: str) -> None:
    assert to_latex(r) == latex


def test_to_python_re_shapes() -> None:
    assert to_python_re(EPS) == "(?:)"
    assert to_python_re(EMPTY) == "(?!)"
    assert to_python_re(union(a, b)) == "(?:a|b)"
    assert to_python_re(star(a)) == "(?:a)*"
    assert to_python_re(sym(".")) == r"\."


# ---------------------------------------------------------------------------
# 4. nullable truth table (and other queries)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("r", "expected"),
    [
        (EMPTY, False),
        (EPS, True),
        (a, False),
        (union(a, b), False),
        (union(a, EPS), True),
        (union(a, star(b)), True),
        (concat(a, b), False),
        (concat(star(a), b), False),
        (concat(star(a), star(b)), True),
        (star(a), True),
        (star(concat(a, b)), True),
    ],
)
def test_nullable(r: Regex, expected: bool) -> None:
    assert nullable(r) is expected


def test_is_empty_language() -> None:
    assert is_empty_language(EMPTY)
    assert not is_empty_language(EPS)
    assert not is_empty_language(a)
    assert not is_empty_language(star(a))
    # Raw (non-canonical) nodes are still answered correctly.
    assert is_empty_language(Concat((a, EMPTY)))
    assert not is_empty_language(Union(frozenset({a, EMPTY})))


def test_alphabet_and_size() -> None:
    r = concat(a, star(union(a, b)), b)
    assert alphabet(r) == frozenset({"a", "b"})
    assert alphabet(EPS) == frozenset()
    # Concat(a, Star(Union(a, b)), b) = 1 + 1 + (1 + (1 + 1 + 1)) + 1
    assert size(r) == 7
    assert size(EMPTY) == 1


# ---------------------------------------------------------------------------
# 5. Simplification rules, one test each (before / after text)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("r", "before", "after"),
    [
        # ε + R·R* = R*
        (union(EPS, concat(a, star(a))), "ε+aa*", "a*"),
        (union(EPS, concat(a, b, star(concat(a, b)))), "ε+ab(ab)*", "(ab)*"),
        # ε + R*·R = R*
        (union(EPS, concat(star(a), a)), "ε+a*a", "a*"),
        (union(EPS, concat(star(concat(a, b)), a, b)), "ε+(ab)*ab", "(ab)*"),
        # ε + R* = R*
        (union(EPS, star(a)), "ε+a*", "a*"),
        # (ε + R)* = R*
        (star(union(EPS, a)), "(ε+a)*", "a*"),
        (star(union(EPS, a, b)), "(ε+a+b)*", "(a+b)*"),
        # R*·R* = R*
        (concat(star(a), star(a)), "a*a*", "a*"),
        (concat(b, star(a), star(a), b), "ba*a*b", "ba*b"),
        # (R*)* = R* (raw node, bypassing the constructor)
        (Star(Star(a)), "(a*)*", "a*"),
        # ε + R = R when R is nullable
        (union(EPS, concat(star(a), star(b))), "ε+a*b*", "a*b*"),
        # Common left factor
        (union(concat(a, b), concat(a, c)), "ab+ac", "a(b+c)"),
        (union(concat(a, b, c), concat(a, b, d)), "abc+abd", "ab(c+d)"),
        # Common right factor
        (union(concat(a, c), concat(b, c)), "ac+bc", "(a+b)c"),
        # Factoring is skipped when it would not strictly shrink the expression
        (union(a, concat(a, b)), "a+ab", "a+ab"),
    ],
)
def test_simplify_rule(r: Regex, before: str, after: str) -> None:
    assert to_text(r) == before
    s = simplify(r)
    assert to_text(s) == after
    assert size(s) <= size(r)


def test_simplify_composite() -> None:
    # (ε + a)*·(ε + aa*)  ->  a*·a*  ->  a*
    r = concat(star(union(EPS, a)), union(EPS, concat(a, star(a))))
    assert to_text(simplify(r)) == "a*"


# ---------------------------------------------------------------------------
# 6. Idempotence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "r",
    [
        union(EPS, concat(a, star(a))),
        union(concat(a, b, c), concat(a, b, d), concat(c, d)),
        star(union(EPS, concat(star(a), star(a)), b)),
        concat(union(EPS, star(a)), union(concat(a, c), concat(b, c))),
    ],
)
def test_simplify_idempotent(r: Regex) -> None:
    once = simplify(r)
    assert simplify(once) == once


# ---------------------------------------------------------------------------
# 7. Property test: simplify preserves the language and never grows
# ---------------------------------------------------------------------------


def regexes(depth: int) -> st.SearchStrategy[Regex]:
    """Random regexes over {a, b} of AST depth <= ``depth`` (built via smart constructors)."""
    leaves = st.sampled_from([EMPTY, EPS, sym("a"), sym("b")])
    if depth == 0:
        return leaves
    sub = regexes(depth - 1)
    return st.one_of(
        leaves,
        st.lists(sub, min_size=2, max_size=3).map(lambda xs: union(*xs)),
        st.lists(sub, min_size=2, max_size=3).map(lambda xs: concat(*xs)),
        sub.map(star),
    )


@settings(max_examples=300, deadline=None)
@given(regexes(4))
def test_simplify_preserves_language(r: Regex) -> None:
    s = simplify(r)
    pr, ps = re.compile(to_python_re(r)), re.compile(to_python_re(s))
    for w in STRINGS_AB_6:
        assert (pr.fullmatch(w) is not None) == (ps.fullmatch(w) is not None), (
            f"{to_text(r)!r} vs simplified {to_text(s)!r} disagree on {w!r}"
        )
    assert size(s) <= size(r)
    assert simplify(s) == s
    assert nullable(s) == nullable(r)


# ---------------------------------------------------------------------------
# 8. to_python_re sanity: known regexes accept / reject known strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("r", "accept", "reject"),
    [
        (concat(a, star(union(a, b)), b), ["ab", "aab", "abab", "abbb"], ["", "a", "b", "ba", "aba"]),
        (star(concat(a, b)), ["", "ab", "abab"], ["a", "aba", "ba"]),
        (concat(star(a), star(b)), ["", "a", "b", "aabb"], ["ba", "aba"]),
        (EPS, [""], ["a"]),
        (EMPTY, [], ["", "a", "ab"]),
        (union(EPS, a), ["", "a"], ["aa", "b"]),
        (concat(sym("."), star(sym("x"))), [".", ".xx"], ["a", "x"]),
    ],
)
def test_python_re_round_trip(r: Regex, accept: list[str], reject: list[str]) -> None:
    for w in accept:
        assert matches(r, w), w
    for w in reject:
        assert not matches(r, w), w
