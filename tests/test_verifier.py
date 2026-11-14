"""Tests for core.verifier: bounded shortlex verification, samples, single strings."""

from __future__ import annotations

from pathlib import Path

import pytest

from core import SolverError
from core.automaton import Automaton, load_automaton, normalize, parse_automaton
from core.equations import build_equations
from core.regex_ast import EPS, concat, star, sym, union
from core.solver import solve
from core.verifier import (
    VerificationResult,
    check_string,
    count_strings,
    sample_strings,
    shortlex_strings,
    verify,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_FILES = sorted(p.name for p in EXAMPLES_DIR.glob("*.json"))

zero, one = sym("0"), sym("1")
a, b = sym("a"), sym("b")
BITS = union(zero, one)


def example(name: str) -> Automaton:
    return load_automaton(EXAMPLES_DIR / name)


def shortlex_key(s: str) -> tuple[int, str]:
    return (len(s), s)


def four_letter_loop() -> Automaton:
    """One final state looping on a, b, c, d: accepts everything over {a,b,c,d}."""
    return parse_automaton(
        {
            "states": ["q0"],
            "alphabet": ["a", "b", "c", "d"],
            "start": "q0",
            "finals": ["q0"],
            "transitions": [{"from": "q0", "symbol": x, "to": "q0"} for x in "abcd"],
        }
    )


# ---------------------------------------------------------------------------
# Full pipeline on every example
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_pipeline_verifies_against_original(filename: str) -> None:
    original = example(filename)
    normalized, _ = normalize(original)
    result = solve(build_equations(normalized))
    v = verify(original, result.regex)
    assert v.passed, (filename, v.counterexample)
    assert v.counterexample is None
    assert v.fa_accepts_counterexample is None and v.re_accepts_counterexample is None
    assert not v.truncated and v.max_length == 8
    assert v.strings_checked == count_strings(len(original.alphabet), 8)


# ---------------------------------------------------------------------------
# Wrong regexes: shortest counterexample and flags
# ---------------------------------------------------------------------------


def test_wrong_regex_re_accepts_too_much() -> None:
    # (0+1)*0 vs "ends in 01": shortlex order is "", "0", ... and "0" is the
    # first string where they differ (the RE accepts it, the FA does not).
    v = verify(example("ends_with_01.json"), concat(star(BITS), zero))
    assert not v.passed
    assert v.counterexample == "0"
    assert v.fa_accepts_counterexample is False
    assert v.re_accepts_counterexample is True
    assert v.strings_checked == 2
    assert v.rejected_samples == ("",)  # agreed on before stopping


def test_wrong_regex_re_accepts_too_little() -> None:
    # (0+1)*101 misses "01", which the FA accepts; "", "0", "1", "00" all agree.
    v = verify(example("ends_with_01.json"), concat(star(BITS), one, zero, one))
    assert v.counterexample == "01"
    assert v.fa_accepts_counterexample is True
    assert v.re_accepts_counterexample is False
    assert v.strings_checked == 5


def test_counterexample_can_be_empty_string() -> None:
    # ε + (0+1)*01 wrongly accepts the empty string.
    v = verify(example("ends_with_01.json"), union(EPS, concat(star(BITS), zero, one)))
    assert v.counterexample == ""
    assert (v.fa_accepts_counterexample, v.re_accepts_counterexample) == (False, True)
    assert v.strings_checked == 1


def test_regex_with_foreign_symbol_raises() -> None:
    with pytest.raises(SolverError, match="not in the automaton's alphabet"):
        verify(example("ends_with_01.json"), concat(star(BITS), sym("2")))
    with pytest.raises(SolverError):
        check_string(example("ends_with_01.json"), sym("x"), "x")


# ---------------------------------------------------------------------------
# Budget truncation
# ---------------------------------------------------------------------------


def test_budget_truncates_default() -> None:
    aut = four_letter_loop()
    v = verify(aut, star(union(*(sym(x) for x in "abcd"))), max_length=12)
    # Σ 4^k: up to 8 -> 87,381 (fits 200,000); up to 9 -> 349,525 (does not)
    assert v.truncated
    assert v.max_length == 8
    assert v.strings_checked == 87_381
    assert v.passed


def test_budget_truncates_small_budget() -> None:
    aut = four_letter_loop()
    v = verify(aut, star(union(*(sym(x) for x in "abcd"))), max_length=12, budget=100)
    assert v.truncated and v.max_length == 3 and v.strings_checked == 85


def test_no_truncation_when_within_budget() -> None:
    v = verify(example("ends_with_01.json"), concat(star(BITS), zero, one), max_length=8)
    assert not v.truncated and v.passed and v.strings_checked == 511


def test_invalid_bounds() -> None:
    aut = example("ends_with_01.json")
    with pytest.raises(ValueError):
        verify(aut, zero, max_length=-1)
    with pytest.raises(ValueError):
        verify(aut, zero, budget=0)


def test_shortlex_enumeration() -> None:
    assert list(shortlex_strings(["b", "a"], 2)) == ["", "a", "b", "aa", "ab", "ba", "bb"]
    assert count_strings(2, 2) == 7
    assert count_strings(3, 0) == 1


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def test_even_zeros_samples() -> None:
    aut = example("even_zeros.json")
    v = verify(aut, star(union(one, concat(zero, star(one), zero))))
    assert v.passed
    assert v.accepted_samples == (
        "", "1", "00", "11", "001", "010", "100", "111", "0000", "0011",
    )
    assert v.rejected_samples == (
        "0", "01", "10", "000", "011", "101", "110", "0001", "0010", "0100",
    )


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_samples_shortlex_and_correct(filename: str) -> None:
    original = example(filename)
    normalized, _ = normalize(original)
    regex = solve(build_equations(normalized)).regex
    v = verify(original, regex)
    for samples, expected in ((v.accepted_samples, True), (v.rejected_samples, False)):
        assert list(samples) == sorted(samples, key=shortlex_key)
        assert len(samples) <= 10
        for s in samples:
            assert original.accepts(s) is expected
            assert check_string(original, regex, s) == (expected, expected)


def test_samples_when_language_is_everything_or_nothing() -> None:
    every = verify(example("single_state_loop.json"), star(union(a, b)))
    assert every.rejected_samples == () and len(every.accepted_samples) == 10
    # ε is wrong for no_finals (which accepts nothing): the counterexample is "".
    nothing = verify(example("no_finals.json"), EPS)
    assert nothing.counterexample == "" and nothing.re_accepts_counterexample is True


def test_sample_strings_for_ui() -> None:
    aut = example("contains_aa.json")
    samples = sample_strings(aut, n=10)
    assert len(samples) == 10
    assert [s for s, _ in samples] == sorted((s for s, _ in samples), key=shortlex_key)
    assert all(aut.accepts(s) is acc for s, acc in samples)
    assert sum(acc for _, acc in samples) == 5  # balanced: 5 accepted, 5 rejected
    # When one kind is missing, the other fills the gap.
    everything = sample_strings(example("single_state_loop.json"), n=6)
    assert len(everything) == 6 and all(acc for _, acc in everything)
    nothing = sample_strings(example("no_finals.json"), n=4)
    assert nothing == (("", False), ("a", False), ("b", False), ("aa", False))
    assert sample_strings(aut, n=0) == ()


# ---------------------------------------------------------------------------
# check_string
# ---------------------------------------------------------------------------


def test_check_string_agrees_with_verify() -> None:
    aut = example("ends_with_01.json")
    wrong = concat(star(BITS), zero)
    v = verify(aut, wrong)
    assert check_string(aut, wrong, v.counterexample) == (
        v.fa_accepts_counterexample,
        v.re_accepts_counterexample,
    )
    right = concat(star(BITS), zero, one)
    for s in ["", "0", "01", "101", "0110", "11101"]:
        fa, rx = check_string(aut, right, s)
        assert fa == rx == aut.accepts(s)


def test_check_string_outside_alphabet() -> None:
    fa, rx = check_string(example("ends_with_01.json"), concat(star(BITS), zero, one), "a01")
    assert (fa, rx) == (False, False)


def test_empty_string_handling() -> None:
    assert check_string(example("even_zeros.json"), star(union(one, concat(zero, star(one), zero))), "") == (True, True)
    assert check_string(example("contains_aa.json"), concat(star(union(a, b)), a, a, star(union(a, b))), "") == (False, False)
    assert verify(example("even_zeros.json"), star(one)).counterexample == "00"


def test_result_is_frozen() -> None:
    v = verify(example("ends_with_01.json"), concat(star(BITS), zero, one))
    assert isinstance(v, VerificationResult)
    with pytest.raises(AttributeError):
        v.passed = False  # type: ignore[misc]
