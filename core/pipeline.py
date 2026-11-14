"""Thin facade over the core pipeline.

Responsibilities: ``run_pipeline(source, order=None) -> PipelineResult``;
also the ``python -m core.pipeline file.json`` CLI.

Stages::

    parse -> normalize -> build_equations -> solve -> verify (against the ORIGINAL)

The UI and the export layer talk to the core only through this module and the
dataclasses it returns. Every failure is raised as a :class:`core.ArdenError`
subclass whose message is written for a student, not a developer.

CLI usage::

    python -m core.pipeline examples/ends_with_01.json [--order q2,q1,q0]
                            [--verify-length 8] [--trace]

Exit codes: 0 verification passed, 1 verification failed, 2 input error.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from core import ArdenError, InvalidAutomatonError, SolverError
from core.automaton import (
    Automaton,
    PreprocessStep,
    load_automaton,
    normalize,
    parse_automaton,
)
from core.equations import EquationSystem, build_equations
from core.regex_ast import to_latex, to_python_re, to_text
from core.solver import SolveResult, StepKind, default_order, solve
from core.verifier import VerificationResult, verify

__all__ = [
    "EXAMPLES_DIR",
    "PipelineResult",
    "run_pipeline",
    "list_examples",
    "format_result",
    "main",
]

#: Directory holding the bundled textbook automata.
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@dataclass(frozen=True)
class PipelineResult:
    """Everything produced by one run of the pipeline.

    Attributes:
        original: the automaton as given by the user.
        normalized: after ε-removal and unreachable/dead-state cleanup.
        preprocessing: the normalization steps (a single "none" step if nothing changed).
        system: the initial equation system built from ``normalized``.
        solve: regex, closed forms, elimination order and the step trace.
        verification: bounded comparison of the RE against ``original``.
        regex_text / regex_latex / regex_python: renderings of the final RE.
        notes: human-readable remarks, e.g. how a custom order was adjusted.
    """

    original: Automaton
    normalized: Automaton
    preprocessing: tuple[PreprocessStep, ...]
    system: EquationSystem
    solve: SolveResult
    verification: VerificationResult
    regex_text: str
    regex_latex: str
    regex_python: str
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _load(source: dict[str, Any] | str | Path | Automaton) -> Automaton:
    """Turn any supported input into an :class:`Automaton`.

    A ``str`` whose first non-blank character is ``{`` is parsed as JSON text;
    any other ``str`` is treated as a file path.
    """
    if isinstance(source, Automaton):
        return source
    if isinstance(source, dict):
        return parse_automaton(source)
    if isinstance(source, Path):
        return load_automaton(source)
    if isinstance(source, str):
        if source.lstrip().startswith("{"):
            return parse_automaton(source)
        return load_automaton(source)
    raise InvalidAutomatonError(
        f"Cannot read an automaton from a value of type {type(source).__name__}; "
        "give a JSON object, JSON text, or the path of a .json file."
    )


def _names(states: Sequence[str]) -> str:
    """Comma-separated state names for notes."""
    return ", ".join(states)


def _adjust_order(
    order: Sequence[str],
    original: Automaton,
    system: EquationSystem,
) -> tuple[tuple[str, ...], list[str]]:
    """Fit a user-supplied order to the normalized system.

    States removed during normalization are dropped; states the order does not
    mention are appended in default order. Unknown or repeated names are
    errors, because they are almost certainly typos.

    Raises:
        SolverError: for names that are not states of the automaton, or repeats.
    """
    order = tuple(s.strip() for s in order)
    known = set(original.states)
    unknown = [s for s in order if s not in known]
    if unknown:
        raise SolverError(
            f"The elimination order mentions {_names(unknown)}, which "
            f"{'is not a state' if len(unknown) == 1 else 'are not states'} of this "
            f"automaton. States: {_names(original.states)}."
        )
    repeated = list(dict.fromkeys(s for i, s in enumerate(order) if s in order[:i]))
    if repeated:
        raise SolverError(
            f"The elimination order lists {_names(repeated)} more than once; "
            "each state may appear only once."
        )
    alive = set(system.state_order)
    removed = [s for s in order if s not in alive]
    kept = [s for s in order if s in alive]
    missing = [s for s in default_order(system) if s not in kept]
    notes: list[str] = []
    if removed:
        notes.append(
            f"State(s) {_names(removed)} were removed during normalization, "
            "so they were dropped from the elimination order."
        )
    if missing:
        notes.append(
            f"State(s) {_names(missing)} were not in the elimination order, "
            "so they were appended in the default order."
        )
    return tuple(kept + missing), notes


def run_pipeline(
    source: dict[str, Any] | str | Path | Automaton,
    order: Sequence[str] | None = None,
    verify_length: int = 8,
) -> PipelineResult:
    """Run parse → normalize → build_equations → solve → verify.

    Args:
        source: a dict, JSON text, a file path (``str`` or ``Path``), or an Automaton.
        order: optional elimination order (see :func:`_adjust_order` for how it
            is fitted to the normalized automaton). ``None`` uses the default
            order: reverse declaration order with the start state last.
        verify_length: strings up to this length are compared (FA vs RE).

    Raises:
        InvalidAutomatonError: unreadable or invalid input.
        SolverError: invalid elimination order or an expression that grows too large.
        ArdenPreconditionError: Arden's theorem would be applied with a nullable P.
        ArdenError: ``verify_length`` is negative.
    """
    if verify_length < 0:
        raise ArdenError(f"The verification length must be 0 or more (got {verify_length}).")
    original = _load(source)
    normalized, steps = normalize(original)
    system = build_equations(normalized)
    notes: list[str] = []
    if order is not None:
        order, notes = _adjust_order(order, original, system)
    solved = solve(system, order)
    verification = verify(original, solved.regex, max_length=verify_length)
    return PipelineResult(
        original=original,
        normalized=normalized,
        preprocessing=tuple(steps),
        system=system,
        solve=solved,
        verification=verification,
        regex_text=to_text(solved.regex),
        regex_latex=to_latex(solved.regex),
        regex_python=to_python_re(solved.regex),
        notes=tuple(notes),
    )


def list_examples(directory: str | Path = EXAMPLES_DIR) -> list[tuple[str, str, str]]:
    """``(filename, name, description)`` for every example automaton, sorted by filename."""
    out: list[tuple[str, str, str]] = []
    for path in sorted(Path(directory).glob("*.json")):
        a = load_automaton(path)
        out.append((path.name, a.name, a.description))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _show(s: str) -> str:
    """Display a string, writing the empty string as ε."""
    return s if s else "ε"


def _indent(text: str, prefix: str = "    ") -> str:
    """Indent every line of ``text`` by ``prefix``."""
    return "\n".join(prefix + line for line in text.splitlines())


def format_result(result: PipelineResult, trace: bool = False) -> str:
    """Human-readable report of a pipeline run (used by the CLI)."""
    a = result.original
    lines = [f"Automaton: {a.name}"]
    if a.description:
        lines.append(f"  {a.description}")
    lines += ["", "== Preprocessing =="]
    lines += [f"- {s.description}" for s in result.preprocessing]
    lines += ["", "== Initial equations ==", result.system.to_text(), ""]
    lines.append(f"== Elimination order: {', '.join(result.solve.order)} ==")
    lines += [f"Note: {n}" for n in result.notes]
    if trace:
        lines += ["", "== Trace =="]
        for step in result.solve.steps:
            lines.append(f"[{step.index}] {step.kind.name}: {step.explanation}")
            if step.kind not in (StepKind.COMBINE_FINALS, StepKind.RESULT):
                lines.append(_indent(step.system_after.to_text()))
    lines += [
        "",
        "== Result ==",
        f"RE:    {result.regex_text}",
        f"LaTeX: {result.regex_latex}",
        "",
        "== Verification ==",
    ]
    v = result.verification
    scope = f"{v.strings_checked} strings up to length {v.max_length}"
    if v.passed:
        lines.append(f"PASSED: the automaton and the RE agree on all {scope}.")
    else:
        fa = "accepts" if v.fa_accepts_counterexample else "rejects"
        rx = "accepts" if v.re_accepts_counterexample else "rejects"
        lines.append(
            f'FAILED: on "{_show(v.counterexample or "")}" the automaton {fa} '
            f"but the RE {rx} (checked {scope})."
        )
    if v.truncated:
        lines.append("(The length was reduced to stay within the string budget.)")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the ``python -m core.pipeline`` CLI."""
    p = argparse.ArgumentParser(
        prog="python -m core.pipeline",
        description="Convert a finite automaton (JSON) to a regular expression with Arden's theorem.",
    )
    p.add_argument("file", help="path to the automaton JSON file")
    p.add_argument("--order", help="comma-separated elimination order, e.g. q2,q1,q0")
    p.add_argument(
        "--verify-length", type=int, default=8,
        help="check all strings up to this length (default: 8)",
    )
    p.add_argument("--trace", action="store_true", help="print every solution step")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns 0 if verification passed, 1 if it failed, 2 on input errors."""
    for stream in (sys.stdout, sys.stderr):
        # Piped output on Windows defaults to a legacy code page that cannot encode ε/∅.
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower() != "utf-8":
            stream.reconfigure(encoding="utf-8")
    args = _build_parser().parse_args(argv)
    order = [s for s in args.order.split(",") if s.strip()] if args.order else None
    try:
        result = run_pipeline(Path(args.file), order=order, verify_length=args.verify_length)
    except ArdenError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(format_result(result, trace=args.trace))
    return 0 if result.verification.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
