"""Finite automaton model.

Responsibilities: the FA model, the JSON schema (pydantic), validation,
epsilon-removal, unreachable/dead-state cleanup, and string simulation.

Data flow::

    dict / JSON text / file
        -> AutomatonSpec   (pydantic v2: schema + convention checks)
        -> Automaton       (frozen domain model used by the rest of the core)
        -> normalize()     (ε-removal, unreachable- and dead-state cleanup)

Every validation failure is reported as :class:`core.InvalidAutomatonError`
with a human-readable message naming the offending item, suitable for display
in the UI.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from core import InvalidAutomatonError
from core.regex_ast import RESERVED_CHARS

__all__ = [
    "EPSILON",
    "EPSILON_ALIASES",
    "TransitionSpec",
    "AutomatonSpec",
    "Automaton",
    "Transition",
    "PreprocessKind",
    "PreprocessStep",
    "parse_automaton",
    "load_automaton",
    "remove_epsilon",
    "remove_unreachable",
    "remove_dead",
    "normalize",
]

#: Canonical spelling of the empty-string label on a transition.
EPSILON = "ε"
#: Spellings accepted in JSON for an ε-transition (normalized to :data:`EPSILON`).
EPSILON_ALIASES: frozenset[str] = frozenset({"ε", "eps"})

#: A transition triple ``(source, symbol, destination)``; ``symbol`` may be ``"ε"``.
Transition = tuple[str, str, str]


# ---------------------------------------------------------------------------
# Input schema (pydantic v2)
# ---------------------------------------------------------------------------


def _duplicates(items: Iterable[str]) -> list[str]:
    """Items occurring more than once, in first-repeat order."""
    seen: set[str] = set()
    dups: list[str] = []
    for x in items:
        if x in seen and x not in dups:
            dups.append(x)
        seen.add(x)
    return dups


class TransitionSpec(BaseModel):
    """One JSON transition ``{"from": ..., "symbol": ..., "to": ...}``.

    ``"eps"`` is accepted as a spelling of ``"ε"`` and normalized to ``"ε"``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    src: str = Field(alias="from")
    symbol: str
    dst: str = Field(alias="to")

    @field_validator("symbol")
    @classmethod
    def _normalize_epsilon(cls, v: str) -> str:
        """Normalize the ε aliases to the canonical ``"ε"``."""
        return EPSILON if v in EPSILON_ALIASES else v


class AutomatonSpec(BaseModel):
    """Pydantic schema of the JSON automaton format documented in CLAUDE.md.

    Besides the core fields, the optional documentation fields ``description``
    and ``expected_language_note`` are allowed. Any other unknown key is
    rejected (this catches typos such as ``"final"`` instead of ``"finals"``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "Untitled automaton"
    description: str = ""
    expected_language_note: str = ""
    states: list[str]
    alphabet: list[str]
    start: str
    finals: list[str] = Field(default_factory=list)
    transitions: list[TransitionSpec] = Field(default_factory=list)

    @field_validator("states")
    @classmethod
    def _check_states(cls, v: list[str]) -> list[str]:
        """States: non-empty list of non-blank, distinct names."""
        if not v:
            raise ValueError("the automaton must have at least one state")
        for s in v:
            if not s.strip():
                raise ValueError(f"state names must not be blank (got {s!r})")
        dups = _duplicates(v)
        if dups:
            raise ValueError(
                "duplicate state name(s): " + ", ".join(repr(d) for d in dups)
            )
        return v

    @field_validator("alphabet")
    @classmethod
    def _check_alphabet(cls, v: list[str]) -> list[str]:
        """Alphabet: non-empty, distinct, single non-reserved characters."""
        if not v:
            raise ValueError("the alphabet must contain at least one symbol")
        for sym in v:
            if len(sym) != 1:
                raise ValueError(
                    f"alphabet symbol {sym!r} is not a single character"
                )
            if sym.isspace():
                raise ValueError(
                    f"alphabet symbol {sym!r} is whitespace, which is not allowed"
                )
            if sym in RESERVED_CHARS:
                raise ValueError(
                    f"alphabet symbol {sym!r} is reserved "
                    "(reserved characters: + * ( ) | ε ∅ and whitespace)"
                )
        dups = _duplicates(v)
        if dups:
            raise ValueError(
                "duplicate alphabet symbol(s): " + ", ".join(repr(d) for d in dups)
            )
        return v

    @field_validator("finals")
    @classmethod
    def _check_finals_distinct(cls, v: list[str]) -> list[str]:
        """Finals must not list the same state twice."""
        dups = _duplicates(v)
        if dups:
            raise ValueError(
                "duplicate final state(s): " + ", ".join(repr(d) for d in dups)
            )
        return v

    @model_validator(mode="after")
    def _check_references(self) -> "AutomatonSpec":
        """Start, finals and transitions must reference declared states/symbols."""
        states = set(self.states)
        symbols = set(self.alphabet)
        problems: list[str] = []
        if self.start not in states:
            problems.append(f"start state {self.start!r} is not in the state list")
        for f in self.finals:
            if f not in states:
                problems.append(f"final state {f!r} is not in the state list")
        for i, t in enumerate(self.transitions):
            where = f"transition #{i} ({t.src!r} --{t.symbol}--> {t.dst!r})"
            if t.src not in states:
                problems.append(f"{where}: unknown source state {t.src!r}")
            if t.dst not in states:
                problems.append(f"{where}: unknown target state {t.dst!r}")
            if t.symbol != EPSILON and t.symbol not in symbols:
                problems.append(
                    f"{where}: symbol {t.symbol!r} is not in the alphabet "
                    "(use \"ε\" or \"eps\" for an epsilon move)"
                )
        if problems:
            raise ValueError("\n".join(problems))
        return self


def _format_loc(loc: tuple[Any, ...]) -> str:
    """Render a pydantic error location as ``transitions[2].from``."""
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += ("." if out else "") + str(part)
    return out


def _humanize(err: ValidationError) -> str:
    """Turn a pydantic ValidationError into a readable, bulleted message."""
    lines: list[str] = []
    for e in err.errors():
        msg = str(e.get("msg", ""))
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        # "...a valid dictionary or instance of TransitionSpec" -> "...a valid dictionary"
        msg = re.sub(r" or instance of \w+", "", msg)
        # Speak JSON, not Python: dictionary -> object, list -> array.
        msg = msg.replace("a valid dictionary", "a JSON object").replace(
            "a valid list", "a JSON array (a list in [ ])"
        )
        loc = _format_loc(tuple(e.get("loc", ())))
        for piece in msg.split("\n"):
            lines.append(f"- {loc}: {piece}" if loc else f"- {piece}")
    return "Invalid automaton:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Automaton:
    """An immutable finite automaton (DFA, NFA or ε-NFA).

    Attributes:
        name: display name.
        states: state names in declaration order (this order is the default
            elimination order used later by the solver).
        alphabet: input symbols in declaration order (never contains ``"ε"``).
        start: the single start state.
        finals: accepting states.
        transitions: ``(src, symbol, dst)`` triples; ``symbol`` may be ``"ε"``.
        description / expected_language_note: optional documentation text.
    """

    name: str
    states: tuple[str, ...]
    alphabet: tuple[str, ...]
    start: str
    finals: frozenset[str]
    transitions: frozenset[Transition]
    description: str = ""
    expected_language_note: str = ""

    def __post_init__(self) -> None:
        """Coerce containers to immutable types and check internal consistency."""
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "alphabet", tuple(self.alphabet))
        object.__setattr__(self, "finals", frozenset(self.finals))
        object.__setattr__(
            self, "transitions", frozenset(tuple(t) for t in self.transitions)
        )
        states = set(self.states)
        if not self.states:
            raise InvalidAutomatonError("the automaton must have at least one state")
        if len(states) != len(self.states):
            raise InvalidAutomatonError(
                "duplicate state name(s): "
                + ", ".join(repr(d) for d in _duplicates(self.states))
            )
        if self.start not in states:
            raise InvalidAutomatonError(f"start state {self.start!r} is not a state")
        if not self.finals <= states:
            raise InvalidAutomatonError(
                "final state(s) "
                + ", ".join(repr(s) for s in sorted(self.finals - states))
                + " are not in the state list"
            )
        if EPSILON in self.alphabet:
            raise InvalidAutomatonError("ε must not be listed in the alphabet")
        for src, symbol, dst in self.transitions:
            if src not in states or dst not in states:
                raise InvalidAutomatonError(
                    f"transition ({src!r}, {symbol!r}, {dst!r}) uses an unknown state"
                )
            if symbol != EPSILON and symbol not in self.alphabet:
                raise InvalidAutomatonError(
                    f"transition ({src!r}, {symbol!r}, {dst!r}) uses an unknown symbol"
                )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: AutomatonSpec) -> "Automaton":
        """Build the domain model from a validated :class:`AutomatonSpec`."""
        return cls(
            name=spec.name,
            states=tuple(spec.states),
            alphabet=tuple(spec.alphabet),
            start=spec.start,
            finals=frozenset(spec.finals),
            transitions=frozenset((t.src, t.symbol, t.dst) for t in spec.transitions),
            description=spec.description,
            expected_language_note=spec.expected_language_note,
        )

    # -- queries --------------------------------------------------------------

    def has_epsilon(self) -> bool:
        """True iff some transition is labelled ``ε``."""
        return any(sym == EPSILON for _, sym, _ in self.transitions)

    def is_deterministic(self) -> bool:
        """True iff there are no ε-moves and at most one move per (state, symbol).

        Completeness (a move for *every* pair) is not required.
        """
        if self.has_epsilon():
            return False
        seen: set[tuple[str, str]] = set()
        for src, sym, _ in self.transitions:
            if (src, sym) in seen:
                return False
            seen.add((src, sym))
        return True

    def step(self, states: Iterable[str], symbol: str) -> frozenset[str]:
        """δ(S, symbol): all states reachable from ``S`` by exactly one ``symbol``-move."""
        current = set(states)
        return frozenset(d for s, x, d in self.transitions if s in current and x == symbol)

    def epsilon_closure(self, states: Iterable[str]) -> frozenset[str]:
        """ε-closure(S): every state reachable from ``S`` using only ε-moves (incl. S)."""
        closure = set(states)
        stack = list(closure)
        while stack:
            s = stack.pop()
            for src, sym, dst in self.transitions:
                if src == s and sym == EPSILON and dst not in closure:
                    closure.add(dst)
                    stack.append(dst)
        return frozenset(closure)

    def accepts(self, word: str) -> bool:
        """Simulate the automaton on ``word`` (subset simulation with ε-closures).

        Works for DFAs, NFAs and ε-NFAs. A character outside the alphabet is
        simply rejected. This is the ground truth used by the verifier.
        """
        current = self.epsilon_closure({self.start})
        for ch in word:
            if ch not in self.alphabet:
                return False
            current = self.epsilon_closure(self.step(current, ch))
            if not current:
                return False
        return bool(current & self.finals)

    # -- export ---------------------------------------------------------------

    def sorted_transitions(self) -> list[Transition]:
        """Transitions in a stable order: by source, symbol (ε first), target declaration order."""
        state_ix = {s: i for i, s in enumerate(self.states)}
        sym_ix = {x: i + 1 for i, x in enumerate(self.alphabet)}
        sym_ix[EPSILON] = 0
        return sorted(
            self.transitions,
            key=lambda t: (state_ix[t[0]], sym_ix[t[1]], state_ix[t[2]]),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict in the CLAUDE.md schema (round-trips through :func:`parse_automaton`)."""
        d: dict[str, Any] = {"name": self.name}
        if self.description:
            d["description"] = self.description
        if self.expected_language_note:
            d["expected_language_note"] = self.expected_language_note
        d.update(
            {
                "states": list(self.states),
                "alphabet": list(self.alphabet),
                "start": self.start,
                "finals": [s for s in self.states if s in self.finals],
                "transitions": [
                    {"from": s, "symbol": x, "to": t}
                    for s, x, t in self.sorted_transitions()
                ],
            }
        )
        return d

    # -- internal helpers ----------------------------------------------------

    def _restrict(self, keep: Iterable[str]) -> "Automaton":
        """Copy keeping only ``keep`` (in original order) and transitions among them."""
        kept = set(keep)
        return Automaton(
            name=self.name,
            states=tuple(s for s in self.states if s in kept),
            alphabet=self.alphabet,
            start=self.start,
            finals=self.finals & kept,
            transitions=frozenset(
                t for t in self.transitions if t[0] in kept and t[2] in kept
            ),
            description=self.description,
            expected_language_note=self.expected_language_note,
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _json_kind(value: Any) -> str:
    """Name a parsed JSON value the way JSON (not Python) does, e.g. ``an array``."""
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    if value is None:
        return "null"
    return f"a {type(value).__name__}"


def parse_automaton(data: dict[str, Any] | str) -> Automaton:
    """Validate a dict or a JSON string and return an :class:`Automaton`.

    Raises:
        InvalidAutomatonError: on malformed JSON or any schema/convention violation.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise InvalidAutomatonError(
                f"Invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
            ) from exc
    if not isinstance(data, dict):
        raise InvalidAutomatonError(
            "Invalid automaton: the JSON must be one object ({ ... }) with the keys "
            f"states, alphabet, start, finals and transitions, not {_json_kind(data)}."
        )
    try:
        spec = AutomatonSpec.model_validate(data)
    except ValidationError as exc:
        raise InvalidAutomatonError(_humanize(exc)) from exc
    return Automaton.from_spec(spec)


def load_automaton(path: str | Path) -> Automaton:
    """Read a UTF-8 JSON file and return the validated :class:`Automaton`.

    Raises:
        InvalidAutomatonError: if the file cannot be read or is invalid.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InvalidAutomatonError(
            f"Cannot read automaton file {str(p)!r}: the file does not exist."
        ) from exc
    except UnicodeDecodeError as exc:
        raise InvalidAutomatonError(
            f"Cannot read automaton file {str(p)!r}: it is not UTF-8 text "
            "(save it as UTF-8 JSON)."
        ) from exc
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise InvalidAutomatonError(f"Cannot read automaton file {str(p)!r}: {reason}.") from exc
    return parse_automaton(text)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

PreprocessKind = Literal["remove_epsilon", "remove_unreachable", "remove_dead", "none"]


@dataclass(frozen=True)
class PreprocessStep:
    """One normalization step, recorded for display in the UI."""

    kind: PreprocessKind
    description: str
    affected_states: tuple[str, ...]
    before: Automaton = field(repr=False)
    after: Automaton = field(repr=False)


def _names(states: Iterable[str]) -> str:
    """Comma-separated state names for messages."""
    return ", ".join(states)


def remove_epsilon(a: Automaton) -> tuple[Automaton, list[PreprocessStep]]:
    """Eliminate ε-transitions with the standard ε-closure construction.

    For every state p and every non-ε symbol x:
        δ'(p, x) = ε-closure(δ(ε-closure(p), x))
    and p is final iff ε-closure(p) contains a final state. The start state
    and the state set are unchanged, so the language is preserved.

    Returns ``(a, [])`` unchanged if ``a`` has no ε-moves.
    """
    if not a.has_epsilon():
        return a, []
    new_transitions: set[Transition] = set()
    new_finals: set[str] = set()
    for p in a.states:
        closure = a.epsilon_closure({p})
        if closure & a.finals:
            new_finals.add(p)
        for x in a.alphabet:
            for q in a.epsilon_closure(a.step(closure, x)):
                new_transitions.add((p, x, q))
    after = Automaton(
        name=a.name,
        states=a.states,
        alphabet=a.alphabet,
        start=a.start,
        finals=frozenset(new_finals),
        transitions=frozenset(new_transitions),
        description=a.description,
        expected_language_note=a.expected_language_note,
    )
    eps_count = sum(1 for t in a.transitions if t[1] == EPSILON)
    eps_sources = {t[0] for t in a.transitions if t[1] == EPSILON}
    newly_final = new_finals - a.finals
    affected = tuple(s for s in a.states if s in eps_sources or s in newly_final)
    desc = (
        f"Removed {eps_count} ε-transition(s) using δ'(p, x) = ε-closure(δ(ε-closure(p), x)) "
        f"for every state p and symbol x."
    )
    if newly_final:
        desc += (
            " State(s) "
            + _names(s for s in a.states if s in newly_final)
            + " became final because their ε-closure contains a final state."
        )
    return after, [PreprocessStep("remove_epsilon", desc, affected, a, after)]


def _reachable_from(a: Automaton, sources: Iterable[str], reverse: bool = False) -> set[str]:
    """States reachable from ``sources`` along any transitions (backwards if ``reverse``)."""
    seen = set(sources)
    queue = deque(seen)
    while queue:
        s = queue.popleft()
        for src, _, dst in a.transitions:
            frm, to = (dst, src) if reverse else (src, dst)
            if frm == s and to not in seen:
                seen.add(to)
                queue.append(to)
    return seen


def remove_unreachable(a: Automaton) -> tuple[Automaton, list[PreprocessStep]]:
    """Drop states that cannot be reached from the start state (and their transitions).

    Unreachable states never influence which strings are accepted, so the
    language is preserved. Returns ``(a, [])`` if every state is reachable.
    """
    reachable = _reachable_from(a, {a.start})
    removed = tuple(s for s in a.states if s not in reachable)
    if not removed:
        return a, []
    after = a._restrict(reachable)
    desc = (
        f"Removed unreachable state(s) {_names(removed)}: no path leads to them "
        f"from the start state {a.start}."
    )
    return after, [PreprocessStep("remove_unreachable", desc, removed, a, after)]


def remove_dead(a: Automaton) -> tuple[Automaton, list[PreprocessStep]]:
    """Drop dead states (no final state reachable from them), except the start state.

    A path through a dead state can never end in acceptance, so removing it
    preserves the language. The start state is always kept: if it is dead the
    language is ∅, and keeping it lets the pipeline still run and yield ∅.
    Returns ``(a, [])`` if no state is removed.
    """
    live = _reachable_from(a, a.finals, reverse=True)
    keep = live | {a.start}
    removed = tuple(s for s in a.states if s not in keep)
    if not removed:
        return a, []
    after = a._restrict(keep)
    desc = (
        f"Removed dead state(s) {_names(removed)}: no final state can be reached from them."
    )
    if a.start not in live:
        desc += (
            f" The start state {a.start} is itself dead (the language is ∅) "
            "but is kept so the conversion can still run."
        )
    return after, [PreprocessStep("remove_dead", desc, removed, a, after)]


def normalize(a: Automaton) -> tuple[Automaton, list[PreprocessStep]]:
    """Bring ``a`` into the form required by the equation builder.

    Runs ε-removal (only if there are ε-moves), then unreachable-state removal,
    then dead-state removal. The relative order of surviving states is kept.
    If nothing changed, returns a single step of kind ``"none"``.
    """
    steps: list[PreprocessStep] = []
    current = a
    for transform in (remove_epsilon, remove_unreachable, remove_dead):
        current, new_steps = transform(current)
        steps.extend(new_steps)
    if not steps:
        steps.append(
            PreprocessStep(
                "none",
                "The automaton is already in the required form: no ε-transitions, "
                "no unreachable states and no dead states.",
                (),
                a,
                a,
            )
        )
    return current, steps
