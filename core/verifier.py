"""Equivalence checking between an automaton and a regular expression.

Responsibilities: bounded string enumeration comparing FA vs RE, and
counterexample reporting.

Verification is deliberately simple: enumerate every string over the original
automaton's alphabet in **shortlex order** (shorter first, then lexicographic)
up to a length bound, and compare ``Automaton.accepts`` (the ground truth)
with ``re.fullmatch`` on the regex's Python rendering. Because of the
shortlex order, the first disagreement found is a *shortest* counterexample.
This is a bounded check, not a proof of equivalence.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterator, Sequence

from core import SolverError
from core.automaton import Automaton
from core.regex_ast import Regex, alphabet, to_python_re, to_text

__all__ = [
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_BUDGET",
    "VerificationResult",
    "count_strings",
    "shortlex_strings",
    "verify",
    "sample_strings",
    "check_string",
]

DEFAULT_MAX_LENGTH = 8
DEFAULT_BUDGET = 200_000
_SAMPLE_COUNT = 10


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of :func:`verify`.

    ``counterexample`` stores the empty string as ``""``; the UI displays it as ``ε``.
    ``strings_checked`` includes the counterexample itself when there is one.
    """

    passed: bool
    max_length: int
    strings_checked: int
    counterexample: str | None
    fa_accepts_counterexample: bool | None
    re_accepts_counterexample: bool | None
    accepted_samples: tuple[str, ...]
    rejected_samples: tuple[str, ...]
    truncated: bool


def count_strings(alphabet_size: int, max_length: int) -> int:
    """Number of strings of length 0..max_length: Σ_{k=0}^{max_length} |Σ|^k."""
    return sum(alphabet_size**k for k in range(max_length + 1))


def shortlex_strings(symbols: Sequence[str], max_length: int) -> Iterator[str]:
    """All strings over ``symbols`` up to ``max_length`` in shortlex order."""
    ordered = sorted(symbols)
    for k in range(max_length + 1):
        for t in itertools.product(ordered, repeat=k):
            yield "".join(t)


def _effective_length(alphabet_size: int, max_length: int, budget: int) -> tuple[int, bool]:
    """Largest length <= max_length whose string count fits in ``budget``; and whether it was cut."""
    if max_length < 0:
        raise ValueError(f"max_length must be >= 0, got {max_length}")
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    length = max_length
    while length > 0 and count_strings(alphabet_size, length) > budget:
        length -= 1
    return length, length < max_length


def _compile(original: Automaton, regex: Regex) -> re.Pattern[str]:
    """Check the regex only uses the automaton's symbols, then compile it once."""
    foreign = alphabet(regex) - set(original.alphabet)
    if foreign:
        raise SolverError(
            f"the regular expression {to_text(regex)} uses symbol(s) "
            f"{', '.join(sorted(foreign))} that are not in the automaton's alphabet "
            f"{{{', '.join(original.alphabet)}}}"
        )
    return re.compile(to_python_re(regex))


def verify(
    original: Automaton,
    regex: Regex,
    max_length: int = DEFAULT_MAX_LENGTH,
    budget: int = DEFAULT_BUDGET,
) -> VerificationResult:
    """Compare ``original`` and ``regex`` on every string up to ``max_length``.

    Strings are enumerated over the original automaton's alphabet in shortlex
    order. If Σ|Σ|^k for k = 0..max_length exceeds ``budget``, the length is
    lowered to the largest value within budget and ``truncated`` is set. The
    scan stops at the first disagreement, which is therefore a shortest
    counterexample.

    Raises:
        SolverError: if the regex uses a symbol outside the automaton's alphabet.
        ValueError: if ``max_length < 0`` or ``budget < 1``.
    """
    pattern = _compile(original, regex)
    length, truncated = _effective_length(len(original.alphabet), max_length, budget)
    accepted: list[str] = []
    rejected: list[str] = []
    checked = 0
    for s in shortlex_strings(original.alphabet, length):
        checked += 1
        fa = original.accepts(s)
        rx = pattern.fullmatch(s) is not None
        if fa != rx:
            return VerificationResult(
                False, length, checked, s, fa, rx,
                tuple(accepted), tuple(rejected), truncated,
            )
        bucket = accepted if fa else rejected
        if len(bucket) < _SAMPLE_COUNT:
            bucket.append(s)
    return VerificationResult(
        True, length, checked, None, None, None,
        tuple(accepted), tuple(rejected), truncated,
    )


def sample_strings(
    automaton: Automaton,
    n: int = _SAMPLE_COUNT,
    max_length: int = DEFAULT_MAX_LENGTH,
    budget: int = DEFAULT_BUDGET,
) -> tuple[tuple[str, bool], ...]:
    """Up to ``n`` short example strings with the automaton's verdict, for a "try a string" panel.

    Takes the shortest accepted and shortest rejected strings (about half
    each; if one kind runs out, the other fills the gap) and returns
    ``(string, accepted)`` pairs in shortlex order.
    """
    if n <= 0:
        return ()
    length, _ = _effective_length(len(automaton.alphabet), max_length, budget)
    accepted: list[str] = []
    rejected: list[str] = []
    for s in shortlex_strings(automaton.alphabet, length):
        (accepted if automaton.accepts(s) else rejected).append(s)
        if len(accepted) >= n and len(rejected) >= n:
            break
    n_acc = min(len(accepted), max((n + 1) // 2, n - len(rejected)))
    n_rej = min(len(rejected), n - n_acc)
    picked = [(s, True) for s in accepted[:n_acc]] + [(s, False) for s in rejected[:n_rej]]
    return tuple(sorted(picked, key=lambda p: (len(p[0]), p[0])))


def check_string(automaton: Automaton, regex: Regex, s: str) -> tuple[bool, bool]:
    """Return ``(automaton accepts s, regex matches s)`` for one user-entered string.

    Raises:
        SolverError: if the regex uses a symbol outside the automaton's alphabet.
    """
    pattern = _compile(automaton, regex)
    return automaton.accepts(s), pattern.fullmatch(s) is not None
