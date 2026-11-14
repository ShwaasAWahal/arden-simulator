"""UI components.

Responsibilities: diagram renderer, equation panel, step controls, and
verification badge.

Everything here is a *pure* helper: no ``st.session_state`` access. The
``render_*`` functions draw Streamlit elements; the others return data
(a graphviz graph, a LaTeX string, a badge) and are unit-tested directly.
The core is reached only through ``core.pipeline`` and the dataclasses it
returns (plus the exception hierarchy in ``core``).
"""

from __future__ import annotations

import json
import re
import traceback
from typing import Any, Iterable, Mapping, Sequence

import graphviz
import streamlit as st

from core import (
    ArdenError,
    ArdenPreconditionError,
    InvalidAutomatonError,
    SolverError,
)
from core.automaton import Automaton  # dataclass type returned by the pipeline
from core.equations import EquationSystem  # dataclass type returned by the pipeline
from core.solver import StepKind, TraceStep
from core.verifier import VerificationResult

__all__ = [
    "FOCUS_COLOR",
    "AFFECTED_COLOR",
    "LATEX_HIGHLIGHT",
    "EPS_DISPLAY",
    "show_string",
    "md_escape",
    "build_diagram",
    "equations_latex",
    "step_badge",
    "parse_json_text",
    "editor_rows",
    "build_manual_spec",
    "render_verification",
    "render_error",
]

#: Fill of the state being solved in the current step (strong amber).
FOCUS_COLOR = "#f59f00"
#: Fill of states whose equations the current step changed (light tint).
AFFECTED_COLOR = "#fff3bf"
#: KaTeX colour for highlighted equations (readable on light and dark backgrounds).
LATEX_HIGHLIGHT = "#e8590c"
#: How the empty string is displayed.
EPS_DISPLAY = "ε"

_INK = "#111111"
_START_NODE = "__start__"


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


def show_string(s: str) -> str:
    """Display a string for students: the empty string becomes ``ε``."""
    return s if s else EPS_DISPLAY


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|~$])")


def md_escape(text: str) -> str:
    """Escape Markdown so regex stars and underscores are shown literally."""
    return _MD_SPECIAL.sub(r"\\\1", text)


# ---------------------------------------------------------------------------
# Diagram
# ---------------------------------------------------------------------------


def build_diagram(
    automaton: Automaton,
    focus_state: str | None = None,
    affected: Iterable[str] = (),
) -> graphviz.Digraph:
    """Graphviz drawing of ``automaton`` (left-to-right).

    * an invisible start node points at the start state;
    * final states are double circles;
    * parallel transitions between the same pair are merged into one edge
      labelled e.g. ``"0, 1"`` (ε first, then alphabet order);
    * ``focus_state`` is filled with :data:`FOCUS_COLOR`, ``affected`` states
      with :data:`AFFECTED_COLOR`.

    A white background with dark ink keeps it readable in light and dark themes.
    """
    affected_set = set(affected)
    g = graphviz.Digraph(
        name=automaton.name or "automaton",
        graph_attr={"rankdir": "LR", "bgcolor": "white", "fontcolor": _INK, "pad": "0.2"},
        node_attr={
            "shape": "circle", "style": "filled", "fillcolor": "white",
            "color": _INK, "fontcolor": _INK, "fontname": "Helvetica",
        },
        edge_attr={"color": "#333333", "fontcolor": _INK, "fontname": "Helvetica"},
    )
    start_node = _START_NODE
    while start_node in automaton.states:
        start_node += "_"
    g.node(start_node, label="", shape="none", width="0", height="0")
    for s in automaton.states:
        attrs: dict[str, str] = {}
        if s in automaton.finals:
            attrs["shape"] = "doublecircle"
        if s == focus_state:
            attrs.update(fillcolor=FOCUS_COLOR, penwidth="2.5")
        elif s in affected_set:
            attrs["fillcolor"] = AFFECTED_COLOR
        g.node(s, label=s, **attrs)
    g.edge(start_node, automaton.start)

    rank = {x: i for i, x in enumerate(automaton.alphabet)}
    labels: dict[tuple[str, str], list[str]] = {}
    for src, symbol, dst in automaton.transitions:
        labels.setdefault((src, dst), []).append(symbol)
    state_rank = {s: i for i, s in enumerate(automaton.states)}
    for (src, dst) in sorted(labels, key=lambda p: (state_rank[p[0]], state_rank[p[1]])):
        symbols = sorted(labels[(src, dst)], key=lambda x: rank.get(x, -1))
        g.edge(src, dst, label=", ".join(symbols))
    return g


# ---------------------------------------------------------------------------
# Equations
# ---------------------------------------------------------------------------


def _colored(tex: str) -> str:
    """Wrap ``tex`` in a scoped KaTeX colour."""
    return "{\\color{" + LATEX_HIGHLIGHT + "}{" + tex + "}}"


def equations_latex(system: EquationSystem, highlight: Iterable[str] = ()) -> str:
    r"""The system as an ``aligned`` block, highlighted equations wrapped in ``\color``.

    Each equation is one line ``lhs &= rhs``; for highlighted states both
    sides are wrapped in ``{\color{...}{...}}`` (scoped, KaTeX-compatible).
    """
    marked = set(highlight)
    lines = []
    for eq in system.equations:
        lhs, rhs = eq.to_latex(system.state_order).split(" = ", 1)
        if eq.lhs in marked:
            lhs, rhs = _colored(lhs), _colored(rhs)
        lines.append(f"{lhs} &= {rhs}")
    return "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}"


# ---------------------------------------------------------------------------
# Step badge
# ---------------------------------------------------------------------------

_KIND_LABEL: Mapping[StepKind, str] = {
    StepKind.INITIAL: "Initial system",
    StepKind.ARDEN: "Arden's theorem",
    StepKind.SUBSTITUTE: "Substitution",
    StepKind.BACK_SUBSTITUTE: "Back-substitution",
    StepKind.COMBINE_FINALS: "Combine final states",
    StepKind.RESULT: "Result",
}
_KIND_COLOR: Mapping[StepKind, str] = {
    StepKind.INITIAL: "gray",
    StepKind.ARDEN: "violet",
    StepKind.SUBSTITUTE: "blue",
    StepKind.BACK_SUBSTITUTE: "orange",
    StepKind.COMBINE_FINALS: "green",
    StepKind.RESULT: "green",
}
_PHASE_LABEL: Mapping[str, str] = {
    "setup": "Setup",
    "elimination": "Elimination",
    "back_substitution": "Back-substitution",
    "result": "Result",
}


def step_badge(step: TraceStep) -> tuple[str, str]:
    """``(label, color)`` for the phase/kind chip; color is a Streamlit Markdown colour name."""
    phase = _PHASE_LABEL.get(step.phase, step.phase)
    kind = _KIND_LABEL[step.kind]
    label = kind if phase == kind else f"{phase} · {kind}"
    return label, _KIND_COLOR[step.kind]


# ---------------------------------------------------------------------------
# Input helpers (upload / manual builder)
# ---------------------------------------------------------------------------


def parse_json_text(text: str) -> dict[str, Any]:
    """Parse uploaded JSON text into a dict, with a student-friendly error.

    Only the JSON syntax is checked here; the automaton itself is validated
    by the pipeline when the simulation runs.

    Raises:
        InvalidAutomatonError: malformed JSON or a top-level value that is not an object.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidAutomatonError(
            f"The file is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."
        ) from exc
    if not isinstance(data, dict):
        raise InvalidAutomatonError(
            "The file must contain one JSON object with the keys "
            "states, alphabet, start, finals and transitions."
        )
    return data


def editor_rows(data: Any) -> list[dict[str, Any]]:
    """Normalize ``st.data_editor`` output (DataFrame, dict of columns or list of rows) to rows."""
    if hasattr(data, "to_dict"):
        return list(data.to_dict("records"))
    if isinstance(data, Mapping):
        columns = list(data)
        length = max((len(data[c]) for c in columns), default=0)
        return [{c: (data[c][i] if i < len(data[c]) else None) for c in columns} for i in range(length)]
    return [dict(r) for r in (data or [])]


def _split(text: str) -> list[str]:
    """Split a comma-separated input into stripped, non-empty items."""
    return [p.strip() for p in (text or "").split(",") if p.strip()]


def build_manual_spec(
    states_text: str,
    alphabet_text: str,
    start: str | None,
    finals: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
    name: str = "Manual automaton",
) -> dict[str, Any]:
    """Assemble a JSON-schema dict from the manual builder's inputs.

    Rows with an empty ``from``, ``symbol`` or ``to`` cell are skipped (the
    editor's blank "new row"); everything else is validated by the pipeline.
    """
    transitions = []
    for row in rows:
        cells = [row.get(k) for k in ("from", "symbol", "to")]
        if any(c is None or (isinstance(c, float) and c != c) or str(c).strip() == "" for c in cells):
            continue
        src, sym, dst = (str(c).strip() for c in cells)
        transitions.append({"from": src, "symbol": sym, "to": dst})
    return {
        "name": name,
        "states": _split(states_text),
        "alphabet": _split(alphabet_text),
        "start": start or "",
        "finals": list(finals),
        "transitions": transitions,
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_verification(result: VerificationResult) -> None:
    """Success/error box summarizing a bounded verification."""
    scope = f"{result.strings_checked:,} strings up to length {result.max_length}"
    if result.passed:
        st.success(f"Verified: the automaton and the regular expression agree on all {scope}.")
    else:
        fa = "accepts" if result.fa_accepts_counterexample else "rejects"
        rx = "accepts" if result.re_accepts_counterexample else "rejects"
        word = md_escape(show_string(result.counterexample or ""))
        st.error(
            f"Mismatch on the string **{word}**: the automaton {fa} it but the "
            f"regular expression {rx} it (shortest counterexample; checked {scope})."
        )
    if result.truncated:
        st.info(
            f"The alphabet is large, so verification was limited to length "
            f"{result.max_length} to stay within the string budget."
        )


_ERROR_TITLES: Sequence[tuple[type[ArdenError], str]] = (
    (InvalidAutomatonError, "The automaton is not valid"),
    (ArdenPreconditionError, "Arden's theorem cannot be applied"),
    (SolverError, "The equations could not be solved"),
    (ArdenError, "There is a problem with the input"),
)


def render_error(exc: BaseException) -> None:
    """Friendly error box.

    ArdenError subclasses (problems in the user's input) show their message;
    anything else is an internal error: a generic message plus a collapsed
    expander with the traceback, so no raw stack trace is shown by default.
    """
    if isinstance(exc, ArdenError):
        title = next(t for cls, t in _ERROR_TITLES if isinstance(exc, cls))
        st.error(f"**{title}.**\n\n{md_escape(str(exc))}")
        return
    st.error(
        "Something went wrong inside the simulator. This is not a problem with your "
        "automaton; please report it with the details below."
    )
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(exc)), language=None)
