"""Tests for core.automaton: schema validation, simulation, preprocessing."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings

from core import ArdenError, InvalidAutomatonError
from core.automaton import (
    EPSILON,
    Automaton,
    load_automaton,
    normalize,
    parse_automaton,
    remove_dead,
    remove_epsilon,
    remove_unreachable,
)
from tests.conftest import random_nfas

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_FILES = sorted(p.name for p in EXAMPLES_DIR.glob("*.json"))


def all_strings(alpha: str | tuple[str, ...], max_len: int) -> list[str]:
    """Every string over ``alpha`` of length 0..max_len."""
    return [
        "".join(t)
        for n in range(max_len + 1)
        for t in itertools.product(alpha, repeat=n)
    ]


def same_language(a1: Automaton, a2: Automaton, max_len: int) -> bool:
    """Exhaustively compare accepts() on all strings over a1's alphabet up to max_len."""
    return all(a1.accepts(w) == a2.accepts(w) for w in all_strings(a1.alphabet, max_len))


def example(name: str) -> Automaton:
    return load_automaton(EXAMPLES_DIR / name)


VALID: dict[str, Any] = {
    "name": "valid",
    "states": ["q0", "q1"],
    "alphabet": ["a", "b"],
    "start": "q0",
    "finals": ["q1"],
    "transitions": [
        {"from": "q0", "symbol": "a", "to": "q1"},
        {"from": "q1", "symbol": "b", "to": "q0"},
    ],
}


def variant(**changes: Any) -> dict[str, Any]:
    """A deep copy of VALID with some top-level keys replaced."""
    d = copy.deepcopy(VALID)
    d.update(changes)
    return d


# ---------------------------------------------------------------------------
# Examples load
# ---------------------------------------------------------------------------


def test_all_ten_examples_present() -> None:
    assert set(EXAMPLE_FILES) == {
        "ends_with_01.json",
        "even_zeros.json",
        "contains_aa.json",
        "div_by_3_binary.json",
        "nfa_a_or_ab.json",
        "epsilon_nfa.json",
        "no_finals.json",
        "unreachable_dead.json",
        "single_state_loop.json",
        "multi_final.json",
    }


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_loads_and_documents_itself(filename: str) -> None:
    a = example(filename)
    assert isinstance(a, Automaton)
    assert a.description and a.expected_language_note
    raw = json.loads((EXAMPLES_DIR / filename).read_text(encoding="utf-8"))
    assert a.states == tuple(raw["states"])  # declaration order preserved


@pytest.mark.parametrize(
    ("filename", "dfa", "eps"),
    [
        ("ends_with_01.json", True, False),
        ("even_zeros.json", True, False),
        ("contains_aa.json", True, False),
        ("div_by_3_binary.json", True, False),
        ("nfa_a_or_ab.json", False, False),
        ("epsilon_nfa.json", False, True),
    ],
)
def test_example_kind(filename: str, dfa: bool, eps: bool) -> None:
    a = example(filename)
    assert a.is_deterministic() is dfa
    assert a.has_epsilon() is eps


def test_div_by_3_name_mentions_empty_string() -> None:
    assert "empty string" in example("div_by_3_binary.json").name


def test_eps_alias_is_normalized() -> None:
    a = example("epsilon_nfa.json")  # uses both "ε" and "eps"
    assert ("q1", EPSILON, "q2") in a.transitions
    assert all(sym != "eps" for _, sym, _ in a.transitions)


def test_parse_from_json_string_and_round_trip() -> None:
    a = parse_automaton(json.dumps(VALID))
    assert a.states == ("q0", "q1")
    assert a.finals == frozenset({"q1"})
    assert parse_automaton(a.to_dict()) == a
    for f in EXAMPLE_FILES:
        e = example(f)
        assert parse_automaton(json.dumps(e.to_dict(), ensure_ascii=False)) == e


# ---------------------------------------------------------------------------
# Invalid input: each rule raises InvalidAutomatonError with a helpful message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "fragments"),
    [
        (variant(states=[], start="q0", finals=[], transitions=[]), ["states", "at least one state"]),
        (variant(states=["q0", "q1", "q0"]), ["duplicate state", "'q0'"]),
        (variant(states=["q0", " "]), ["blank"]),
        (variant(alphabet=[]), ["alphabet", "at least one symbol"]),
        (variant(alphabet=["a", "b", "a"]), ["duplicate alphabet symbol", "'a'"]),
        (variant(alphabet=["a", "ab"]), ["'ab'", "not a single character"]),
        (variant(alphabet=["a", ""]), ["''", "not a single character"]),
        (variant(alphabet=["a", "b", "+"]), ["'+'", "reserved"]),
        (variant(alphabet=["a", "b", "*"]), ["'*'", "reserved"]),
        (variant(alphabet=["a", "b", "("]), ["'('", "reserved"]),
        (variant(alphabet=["a", "b", ")"]), ["')'", "reserved"]),
        (variant(alphabet=["a", "b", "|"]), ["'|'", "reserved"]),
        (variant(alphabet=["a", "b", "ε"]), ["'ε'", "reserved"]),
        (variant(alphabet=["a", "b", "∅"]), ["'∅'", "reserved"]),
        (variant(alphabet=["a", "b", " "]), ["whitespace"]),
        (variant(alphabet=["a", "b", "\n"]), ["whitespace"]),
        (variant(start="q9"), ["start state 'q9'", "not in the state list"]),
        (variant(finals=["q1", "qX"]), ["final state 'qX'", "not in the state list"]),
        (variant(finals=["q1", "q1"]), ["duplicate final", "'q1'"]),
        (
            variant(transitions=[{"from": "qZ", "symbol": "a", "to": "q1"}]),
            ["transition #0", "unknown source state 'qZ'"],
        ),
        (
            variant(transitions=[{"from": "q0", "symbol": "a", "to": "q1"},
                                 {"from": "q0", "symbol": "a", "to": "qY"}]),
            ["transition #1", "unknown target state 'qY'"],
        ),
        (
            variant(transitions=[{"from": "q0", "symbol": "c", "to": "q1"}]),
            ["transition #0", "symbol 'c' is not in the alphabet"],
        ),
        (
            variant(transitions=[{"from": "q0", "symbol": "a"}]),
            ["transitions[0].to", "Field required"],
        ),
        (variant(final=["q1"]), ["final", "Extra inputs are not permitted"]),
        ({k: v for k, v in VALID.items() if k != "start"}, ["start", "Field required"]),
    ],
)
def test_invalid_input_rejected(data: dict[str, Any], fragments: list[str]) -> None:
    with pytest.raises(InvalidAutomatonError) as info:
        parse_automaton(data)
    msg = str(info.value)
    assert msg.startswith("Invalid automaton:")
    for frag in fragments:
        assert frag in msg, f"{frag!r} missing from message:\n{msg}"
    assert "Value error," not in msg  # pydantic noise is stripped


def test_multiple_reference_problems_all_reported() -> None:
    data = variant(
        start="nope",
        transitions=[{"from": "q0", "symbol": "z", "to": "q7"}],
    )
    with pytest.raises(InvalidAutomatonError) as info:
        parse_automaton(data)
    msg = str(info.value)
    assert "'nope'" in msg and "'z'" in msg and "'q7'" in msg
    assert msg.count("\n- ") >= 3


def test_invalid_json_text() -> None:
    with pytest.raises(InvalidAutomatonError, match="Invalid JSON"):
        parse_automaton("{not json")


def test_json_not_an_object() -> None:
    with pytest.raises(InvalidAutomatonError, match="expected a JSON object"):
        parse_automaton("[1, 2]")


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InvalidAutomatonError, match="Cannot read"):
        load_automaton(tmp_path / "missing.json")


def test_errors_are_arden_errors() -> None:
    with pytest.raises(ArdenError):
        parse_automaton(variant(states=[]))


def test_domain_model_rejects_inconsistent_construction() -> None:
    with pytest.raises(InvalidAutomatonError):
        Automaton("x", ("q0",), ("a",), "q1", frozenset(), frozenset())
    with pytest.raises(InvalidAutomatonError):
        Automaton("x", ("q0",), ("a",), "q0", frozenset(), frozenset({("q0", "b", "q0")}))


# ---------------------------------------------------------------------------
# accepts(): hand-written accept / reject lists for each example
# ---------------------------------------------------------------------------

ACCEPT_REJECT: dict[str, tuple[list[str], list[str]]] = {
    "ends_with_01.json": (
        ["01", "001", "101", "1101", "0101", "11001"],
        ["", "0", "1", "10", "011", "010"],
    ),
    "even_zeros.json": (
        ["", "1", "00", "11", "1001", "0110", "000111000"],
        ["0", "10", "01", "000", "1011", "01010"],
    ),
    "contains_aa.json": (
        ["aa", "baa", "aab", "abaab", "bbaabb", "aaaa"],
        ["", "a", "b", "ab", "aba", "babab"],
    ),
    "div_by_3_binary.json": (
        ["", "0", "11", "110", "1001", "1100", "1111", "10010"],  # 0,0,3,6,9,12,15,18
        ["1", "10", "100", "101", "111", "1000"],  # 1,2,4,5,7,8
    ),
    "nfa_a_or_ab.json": (
        ["", "a", "ab", "aab", "aba", "abab", "aaab"],
        ["b", "ba", "abb", "bb", "aabb", "bab"],
    ),
    "epsilon_nfa.json": (
        ["", "a", "b", "c", "abc", "aabbcc", "ac", "bbc"],
        ["ba", "cb", "ca", "abca", "acb", "cab"],
    ),
    "no_finals.json": (
        [],
        ["", "a", "b", "ab", "ba", "aab"],
    ),
    "unreachable_dead.json": (
        ["a", "aa", "aaa", "aaaa", "aaaaa"],
        ["", "b", "ab", "ba", "aab", "abb"],
    ),
    "single_state_loop.json": (
        ["", "a", "b", "ab", "ba", "abba", "bbbbb"],
        # Every string over {a, b} is accepted; only foreign symbols are rejected.
        ["c", "ac", "abc", "x", "ε"],
    ),
    "multi_final.json": (
        ["a", "bb", "ba", "abb", "bbb", "aba"],
        ["", "b", "ab", "bab", "abab", "aab"],
    ),
}


def test_every_example_has_accept_reject_lists() -> None:
    assert set(ACCEPT_REJECT) == set(EXAMPLE_FILES)


@pytest.mark.parametrize("filename", sorted(ACCEPT_REJECT))
def test_accepts(filename: str) -> None:
    a = example(filename)
    accept, reject = ACCEPT_REJECT[filename]
    for w in accept:
        assert a.accepts(w), f"{filename} should accept {w!r}"
    for w in reject:
        assert not a.accepts(w), f"{filename} should reject {w!r}"


def test_epsilon_closure() -> None:
    a = example("epsilon_nfa.json")
    assert a.epsilon_closure({"q0"}) == frozenset({"q0", "q1", "q2"})
    assert a.epsilon_closure({"q1"}) == frozenset({"q1", "q2"})
    assert a.epsilon_closure({"q2"}) == frozenset({"q2"})
    assert a.epsilon_closure(set()) == frozenset()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_remove_epsilon_preserves_language() -> None:
    a = example("epsilon_nfa.json")
    b, steps = remove_epsilon(a)
    assert not b.has_epsilon()
    assert b.start == a.start and b.states == a.states
    assert same_language(a, b, 7)
    assert len(steps) == 1 and steps[0].kind == "remove_epsilon"
    assert steps[0].before is a and steps[0].after is b
    # q0 and q1 reach the final q2 via ε, so they become final.
    assert b.finals == frozenset({"q0", "q1", "q2"})
    assert "q0" in steps[0].affected_states and "q1" in steps[0].affected_states


def test_remove_epsilon_noop_without_epsilon() -> None:
    a = example("ends_with_01.json")
    assert remove_epsilon(a) == (a, [])


def test_remove_unreachable() -> None:
    a = example("unreachable_dead.json")
    b, steps = remove_unreachable(a)
    assert b.states == ("q0", "q1", "q2")
    assert all("q3" not in (t[0], t[2]) for t in b.transitions)
    assert [s.kind for s in steps] == ["remove_unreachable"]
    assert steps[0].affected_states == ("q3",)
    assert "q3" in steps[0].description
    assert same_language(a, b, 7)


def test_remove_dead() -> None:
    a = example("unreachable_dead.json")
    b, steps = remove_dead(a)
    # q3 can still reach the final q1, so only q2 is dead.
    assert b.states == ("q0", "q1", "q3")
    assert all("q2" not in (t[0], t[2]) for t in b.transitions)
    assert [s.kind for s in steps] == ["remove_dead"]
    assert steps[0].affected_states == ("q2",)
    assert same_language(a, b, 7)


def test_remove_dead_keeps_dead_start() -> None:
    a = example("no_finals.json")
    b, steps = remove_dead(a)
    assert b.states == ("q0",)
    assert b.finals == frozenset()
    assert steps[0].affected_states == ("q1",)
    assert "start state q0 is itself dead" in steps[0].description
    assert same_language(a, b, 7)


def test_preprocessing_preserves_state_order() -> None:
    # Declared order deliberately not alphabetical; "x" is unreachable, "d" dead.
    a = parse_automaton(
        {
            "states": ["z", "x", "m", "d", "a"],
            "alphabet": ["0", "1"],
            "start": "m",
            "finals": ["a"],
            "transitions": [
                {"from": "m", "symbol": "0", "to": "z"},
                {"from": "z", "symbol": "1", "to": "a"},
                {"from": "m", "symbol": "1", "to": "d"},
                {"from": "x", "symbol": "0", "to": "a"},
                {"from": "a", "symbol": "eps", "to": "m"},
            ],
        }
    )
    b, steps = normalize(a)
    assert b.states == ("z", "m", "a")
    assert [s.kind for s in steps] == ["remove_epsilon", "remove_unreachable", "remove_dead"]
    assert same_language(a, b, 7)


def test_normalize_unreachable_dead_example() -> None:
    a = example("unreachable_dead.json")
    b, steps = normalize(a)
    assert b.states == ("q0", "q1")
    assert [s.kind for s in steps] == ["remove_unreachable", "remove_dead"]
    assert steps[0].after is steps[1].before  # steps chain together
    assert same_language(a, b, 7)


def test_normalize_clean_dfa_returns_single_none_step() -> None:
    a = example("ends_with_01.json")
    b, steps = normalize(a)
    assert b == a
    assert len(steps) == 1
    assert steps[0].kind == "none"
    assert steps[0].affected_states == ()
    assert "already" in steps[0].description


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_normalize_examples_preserve_language(filename: str) -> None:
    a = example(filename)
    b, _ = normalize(a)
    assert not b.has_epsilon()
    assert same_language(a, b, 7)


# ---------------------------------------------------------------------------
# Property: normalize preserves the language of random small NFAs
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(random_nfas())
def test_normalize_preserves_language_property(a: Automaton) -> None:
    b, steps = normalize(a)
    assert not b.has_epsilon()
    assert b.start == a.start
    assert same_language(a, b, 6)
    # surviving states keep their relative order
    assert list(b.states) == [s for s in a.states if s in set(b.states)]
    assert steps and all(s.kind in {"remove_epsilon", "remove_unreachable", "remove_dead", "none"} for s in steps)


# ---------------------------------------------------------------------------
# Review additions: file errors, message cleanup, domain-model checks
# ---------------------------------------------------------------------------


def test_load_non_utf8_file(tmp_path: Path) -> None:
    p = tmp_path / "latin1.json"
    p.write_bytes(b"\xff\xfe{}")
    with pytest.raises(InvalidAutomatonError, match="not UTF-8"):
        load_automaton(p)


def test_load_file_errors_are_readable(tmp_path: Path) -> None:
    with pytest.raises(InvalidAutomatonError) as missing:
        load_automaton(tmp_path / "nope.json")
    assert "does not exist" in str(missing.value) and "Errno" not in str(missing.value)
    with pytest.raises(InvalidAutomatonError, match="Cannot read"):
        load_automaton(tmp_path)  # a directory


def test_pydantic_model_names_hidden() -> None:
    with pytest.raises(InvalidAutomatonError) as info:
        parse_automaton(variant(transitions=[1]))
    msg = str(info.value)
    assert "transitions[0]: Input should be a valid dictionary" in msg
    assert "TransitionSpec" not in msg


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"states": ()}, "at least one state"),
        ({"states": ("q0", "q0")}, "duplicate state name(s): 'q0'"),
        ({"finals": frozenset({"x"})}, "final state(s) 'x' are not in the state list"),
        ({"alphabet": ("a", "ε")}, "ε must not be listed"),
        ({"transitions": frozenset({("q0", "a", "q9")})}, "unknown state"),
    ],
)
def test_domain_model_consistency_checks(changes: dict[str, Any], fragment: str) -> None:
    args: dict[str, Any] = dict(
        name="x", states=("q0",), alphabet=("a",), start="q0",
        finals=frozenset(), transitions=frozenset(),
    )
    args.update(changes)
    with pytest.raises(InvalidAutomatonError) as info:
        Automaton(**args)
    assert fragment in str(info.value)
