"""Report builder.

Responsibilities: builds Markdown and LaTeX reports from a PipelineResult.

Both reports have the same sections (see :data:`SECTIONS`). The Python side
prepares every value (escaped for the target format) and the Jinja2
templates in ``export/templates/`` only lay them out. The LaTeX environment
uses ``\\BLOCK{...}`` / ``\\VAR{...}`` / ``\\#{...}`` delimiters so template
syntax never clashes with LaTeX braces, and autoescaping is off for both.

This package may import from ``core`` but never from ``ui``.
"""

from __future__ import annotations

import datetime as _dt
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import jinja2

from core.automaton import EPSILON, Automaton
from core.equations import EquationSystem, state_to_latex
from core.pipeline import PipelineResult
from core.solver import StepKind

__all__ = [
    "TEMPLATES_DIR",
    "SECTIONS",
    "render_markdown",
    "render_latex",
    "slugify",
    "latex_escape",
    "md_escape",
]

#: Directory holding ``report.md.j2`` and ``report.tex.j2``.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

#: Section titles, in order, shared by both report formats.
SECTIONS: tuple[str, ...] = (
    "Arden's Theorem",
    "Input automaton",
    "Preprocessing",
    "Initial equation system",
    "Elimination order",
    "Solution steps",
    "Final regular expression",
    "Verification",
)

_KIND_LABEL = {
    StepKind.INITIAL: "Initial system",
    StepKind.ARDEN: "Arden's theorem",
    StepKind.SUBSTITUTE: "Substitution",
    StepKind.BACK_SUBSTITUTE: "Back-substitution",
    StepKind.COMBINE_FINALS: "Combine final states",
    StepKind.RESULT: "Result",
}
_PHASE_LABEL = {
    "setup": "setup",
    "elimination": "elimination",
    "back_substitution": "back-substitution",
    "result": "result",
}


# ---------------------------------------------------------------------------
# Escaping and small helpers
# ---------------------------------------------------------------------------

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "#": r"\#",
    "$": r"\$",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
    # Long regexes in prose have no spaces; let TeX break a line after a union.
    "+": r"+\allowbreak{}",
}
# Non-ASCII characters that appear in core messages, mapped to math-mode commands
# so the document compiles with plain pdflatex.
_LATEX_UNICODE = {
    "ε": r"$\varepsilon$",
    "∅": r"$\varnothing$",
    "·": r"$\cdot$",
    "→": r"$\rightarrow$",
    "⇒": r"$\Rightarrow$",
    "δ": r"$\delta$",
    "★": r"$\star$",
    "…": r"\ldots{}",
    "–": "--",
    "—": "---",
}


def latex_escape(text: str) -> str:
    """Escape plain text for LaTeX (special characters and the Unicode used by the core)."""
    return "".join(_LATEX_SPECIAL.get(c) or _LATEX_UNICODE.get(c) or c for c in text)


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|~$])")


def md_escape(text: str) -> str:
    """Escape Markdown so regex stars and underscores are shown literally."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def slugify(name: str) -> str:
    """File-name-safe slug of an automaton name, e.g. ``"ends in a or in bb"`` → ``"ends-in-a-or-in-bb"``."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug or "automaton"


def _show(s: str) -> str:
    """The empty string is displayed as ε."""
    return s if s else EPSILON


def _symbol_columns(a: Automaton) -> list[str]:
    """Transition-table columns: the alphabet, plus ε if the automaton has ε-moves."""
    return [*a.alphabet, EPSILON] if a.has_epsilon() else list(a.alphabet)


def _transition_rows(a: Automaton) -> list[dict[str, Any]]:
    """One row per state: start/final markers and the target set for each symbol."""
    rows = []
    for s in a.states:
        cells = []
        for x in _symbol_columns(a):
            targets = [d for d in a.states if (s, x, d) in a.transitions]
            cells.append(targets)
        rows.append(
            {"state": s, "start": s == a.start, "final": s in a.finals, "targets": cells}
        )
    return rows


def _align_block(system: EquationSystem) -> str:
    """Body of an ``align*`` environment: one ``lhs &= rhs`` line per equation."""
    lines = []
    for eq in system.equations:
        lhs, rhs = eq.to_latex(system.state_order).split(" = ", 1)
        lines.append(f"  {lhs} &= {rhs}")
    return " \\\\\n".join(lines)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def _context(result: PipelineResult, today: _dt.date | None) -> dict[str, Any]:
    """Format-neutral data shared by both templates (raw strings; filters escape)."""
    a = result.original
    v = result.verification
    return {
        "sections": SECTIONS,
        "name": a.name,
        "description": a.description,
        "date": (today or _dt.date.today()).isoformat(),
        "automaton": a,
        "columns": _symbol_columns(a),
        "rows": _transition_rows(a),
        "finals": [s for s in a.states if s in a.finals],
        "preprocessing": result.preprocessing,
        "system": result.system,
        "system_block": _align_block(result.system),
        "order": result.solve.order,
        "notes": result.notes,
        "steps": [
            {
                "number": s.index + 1,
                "kind": _KIND_LABEL[s.kind],
                "phase": _PHASE_LABEL.get(s.phase, s.phase),
                "focus": s.focus_state,
                "explanation": s.explanation,
                "rule_latex": s.rule_latex,
                "show_system": s.kind not in (StepKind.COMBINE_FINALS, StepKind.RESULT),
                "system_text": s.system_after.to_text(),
                "system_block": _align_block(s.system_after),
            }
            for s in result.solve.steps
        ],
        "regex_text": result.regex_text,
        "regex_latex": result.regex_latex,
        "verification": v,
        "accepted": [_show(s) for s in v.accepted_samples],
        "rejected": [_show(s) for s in v.rejected_samples],
    }


# ---------------------------------------------------------------------------
# Jinja environments
# ---------------------------------------------------------------------------


def _common_filters(env: jinja2.Environment, filters: dict[str, Callable[..., str]]) -> jinja2.Environment:
    """Register ``filters`` on ``env`` and return it."""
    env.filters.update(filters)
    return env


@lru_cache(maxsize=1)
def _markdown_env() -> jinja2.Environment:
    """Plain Jinja2 environment for the Markdown template."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    return _common_filters(env, {"md": md_escape})


def _state_math(name: str) -> str:
    """A state name in LaTeX math (``q12`` → ``q_{12}``, other names → ``\\mathit{...}``)."""
    return state_to_latex(name)


def _state_set_math(states: Sequence[str]) -> str:
    """A target set as LaTeX math, e.g. ``\\{q_{1}, q_{2}\\}``; empty → ``\\varnothing``."""
    if not states:
        return r"\varnothing"
    return r"\{" + ", ".join(state_to_latex(s) for s in states) + r"\}"


@lru_cache(maxsize=1)
def _latex_env() -> jinja2.Environment:
    """Jinja2 environment with LaTeX-friendly delimiters (\\BLOCK, \\VAR, \\#)."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
        line_statement_prefix=None,
        line_comment_prefix="%#",
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    return _common_filters(
        env,
        {
            "tex": latex_escape,
            "state": _state_math,
            "stateset": _state_set_math,
            "inline_math": lambda m: f"${m}$",
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_markdown(result: PipelineResult, today: _dt.date | None = None) -> str:
    """Markdown report of a pipeline run (sections listed in :data:`SECTIONS`)."""
    return _markdown_env().get_template("report.md.j2").render(**_context(result, today))


def render_latex(result: PipelineResult, today: _dt.date | None = None) -> str:
    """Complete, compilable LaTeX (article) report of a pipeline run.

    Uses only amsmath, amssymb and booktabs; equation systems are ``align*``
    environments and the automaton is a booktabs transition table.
    """
    return _latex_env().get_template("report.tex.j2").render(**_context(result, today))
