"""Tests for export.report: Markdown and LaTeX reports, JSON round-trip, pdflatex."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from core.automaton import parse_automaton
from core.pipeline import PipelineResult, run_pipeline
from core.regex_ast import EPS, concat, star, sym, union
from core.verifier import verify
from export.report import (
    SECTIONS,
    latex_escape,
    md_escape,
    render_latex,
    render_markdown,
    slugify,
)

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
EXAMPLE_FILES = sorted(p.name for p in EXAMPLES_DIR.glob("*.json"))

# The LaTeX environment's delimiters ("}}" alone is legitimate LaTeX, e.g. nested groups),
# plus default Jinja statement/comment openers that would mean a delimiter mix-up.
JINJA_MARKERS = ("\\VAR{", "\\BLOCK{", "\\#{", "{%", "{#")


def result_for(name: str, **kwargs: object) -> PipelineResult:
    return run_pipeline(EXAMPLES_DIR / name, **kwargs)  # type: ignore[arg-type]


def braces_balanced(tex: str) -> bool:
    """Unescaped { and } nest correctly (a backslash escapes the next character)."""
    depth = 0
    i = 0
    while i < len(tex):
        c = tex[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


# ---------------------------------------------------------------------------
# Every example
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_markdown_report(filename: str) -> None:
    r = result_for(filename)
    md = render_markdown(r)
    assert md.strip()
    assert r.regex_text in md
    assert md.count("\n### Step ") == len(r.solve.steps)
    assert not any(m in md for m in ("{{", "{%", "{#"))


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_latex_report(filename: str) -> None:
    r = result_for(filename)
    tex = render_latex(r)
    assert r.regex_latex in tex
    assert tex.count("\\subsection*{Step ") == len(r.solve.steps)
    assert tex.lstrip().startswith("\\documentclass")
    assert "\\begin{document}" in tex and tex.rstrip().endswith("\\end{document}")
    for pkg in ("amsmath", "amssymb", "booktabs"):
        assert f"\\usepackage{{{pkg}}}" in tex
    assert braces_balanced(tex)
    leftovers = [m for m in JINJA_MARKERS if m in tex]
    assert not leftovers, leftovers
    systems_shown = 1 + sum(
        1 for s in r.solve.steps if s.kind.name not in ("COMBINE_FINALS", "RESULT")
    )
    assert tex.count("\\begin{align*}") == systems_shown
    assert "\\begin{tabular}" in tex and "\\toprule" in tex


def test_markdown_sections_in_order() -> None:
    md = render_markdown(result_for("ends_with_01.json"))
    assert md.startswith("# strings ending in 01\n")
    positions = [md.index(f"\n## {title}\n") for title in SECTIONS]
    assert positions == sorted(positions)
    assert len(SECTIONS) == 8


def test_latex_sections_in_order() -> None:
    tex = render_latex(result_for("ends_with_01.json"))
    positions = [tex.index(f"\\section{{{latex_escape(title)}}}") for title in SECTIONS]
    assert positions == sorted(positions)


def test_generation_date() -> None:
    r = result_for("ends_with_01.json")
    day = dt.date(2026, 1, 2)
    assert "Generated on 2026-01-02" in render_markdown(r, today=day)
    assert "\\date{2026-01-02}" in render_latex(r, today=day)
    assert dt.date.today().isoformat() in render_markdown(r)


# ---------------------------------------------------------------------------
# Section content
# ---------------------------------------------------------------------------


def test_transition_table_markdown() -> None:
    md = render_markdown(result_for("ends_with_01.json"))
    assert "| State | 0 | 1 |" in md
    assert "| → q0 | \\{q1\\} | \\{q0\\} |" in md
    assert "| \\* q2 | \\{q1\\} | \\{q0\\} |" in md


def test_transition_table_latex_and_epsilon_column() -> None:
    tex = render_latex(result_for("epsilon_nfa.json"))
    assert "\\begin{tabular}{lcccc}" in tex  # state + a, b, c, ε
    assert "State & a & b & c & $\\varepsilon$ \\\\" in tex
    assert "$\\rightarrow\\,q_{0}$ & $\\{q_{0}\\}$ & $\\varnothing$" in tex
    md = render_markdown(result_for("epsilon_nfa.json"))
    assert "| State | a | b | c | ε |" in md


def test_start_and_final_marker_on_same_state() -> None:
    md = render_markdown(result_for("even_zeros.json"))
    assert "| → \\* even |" in md
    assert "$\\rightarrow\\,{*}\\,\\mathit{even}$" in render_latex(result_for("even_zeros.json"))


def test_preprocessing_and_order_and_notes() -> None:
    r = result_for("unreachable_dead.json", order=["q3", "q2", "q1", "q0"])
    md = render_markdown(r)
    tex = render_latex(r)
    assert "Removed unreachable state(s) q3" in md
    assert "q1 → q0" in md
    assert "> State(s) q3, q2 were removed during normalization" in md
    assert "\\textit{Note:} State(s) q3, q2 were removed" in tex
    clean = render_markdown(result_for("ends_with_01.json"))
    assert "already in the required form" in clean


def test_verification_samples_show_epsilon() -> None:
    r = result_for("even_zeros.json")
    md = render_markdown(r)
    assert "**Passed:** the automaton and the regular expression agree on all 511 strings up to length 8." in md
    assert "Accepted by both (shortest first):** ε, 1, 00, 11" in md
    tex = render_latex(r)
    assert "$\\varepsilon$, 1, 00, 11" in tex


def test_failed_verification_rendered() -> None:
    r = result_for("ends_with_01.json")
    wrong = verify(r.original, union(EPS, concat(star(union(sym("0"), sym("1"))), sym("0"), sym("1"))))
    bad = dataclasses.replace(r, verification=wrong)
    md = render_markdown(bad)
    assert "**Failed:** on the string `ε` the automaton rejects it but the regular expression accepts it" in md
    tex = render_latex(bad)
    assert "\\textbf{Failed:} on the string \\texttt{$\\varepsilon$}" in tex
    assert braces_balanced(tex)


def test_truncation_mentioned() -> None:
    r = result_for("epsilon_nfa.json", verify_length=20)
    assert r.verification.truncated
    assert "reduced to" in render_markdown(r) and "reduced to" in render_latex(r)


def test_no_finals_report() -> None:
    r = result_for("no_finals.json")
    md = render_markdown(r)
    assert "**Final states:** none" in md
    assert "`∅`" in md
    assert "\\textbf{Final states:} none" in render_latex(r)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("strings ending in 01", "strings-ending-in-01"),
        ("ε-NFA for a*b*c*", "nfa-for-a-b-c"),
        ("binary numbers divisible by 3 (empty string accepted, read as 0)",
         "binary-numbers-divisible-by-3-empty-string-accepted-read-as-0"),
        ("Ünïcödé Name", "unicode-name"),
        ("εεε", "automaton"),
        ("", "automaton"),
    ],
)
def test_slugify(name: str, slug: str) -> None:
    assert slugify(name) == slug


def test_latex_escape() -> None:
    assert latex_escape("a_b & 50% #1 {x} $") == r"a\_b \& 50\% \#1 \{x\} \$"
    assert latex_escape("ε ∅ q0·a → ⇒") == (
        r"$\varepsilon$ $\varnothing$ q0$\cdot$a $\rightarrow$ $\Rightarrow$"
    )
    assert latex_escape("a+b") == r"a+\allowbreak{}b"
    assert latex_escape("\\^~") == r"\textbackslash{}\textasciicircum{}\textasciitilde{}"


def test_md_escape() -> None:
    assert md_escape("(a+b)*_x") == r"(a+b)\*\_x"


def test_export_never_imports_ui() -> None:
    for path in (ROOT / "export").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+ui\b", source, re.M), path


# ---------------------------------------------------------------------------
# JSON round-trip (the UI's "Download automaton JSON")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_json_round_trip(filename: str) -> None:
    r = result_for(filename)
    downloaded = json.dumps(r.original.to_dict(), indent=2, ensure_ascii=False)
    reparsed = parse_automaton(downloaded)
    assert reparsed == r.original
    assert run_pipeline(json.loads(downloaded)).regex_text == r.regex_text


# ---------------------------------------------------------------------------
# Real compilation (only if pdflatex is installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not on PATH")
def test_pdflatex_compiles(tmp_path: Path) -> None:
    tex_file = tmp_path / "ends_with_01.tex"
    tex_file.write_text(render_latex(result_for("ends_with_01.json")), encoding="utf-8")
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
        cwd=tmp_path,
        capture_output=True,
        timeout=300,
    )
    log = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode == 0, log[-2000:]
    assert (tmp_path / "ends_with_01.pdf").stat().st_size > 0
