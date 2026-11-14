"""Streamlit entry point (``streamlit run ui/app.py``).

Interactive simulator: pick or build a finite automaton, convert it to a
regular expression with Arden's theorem, and step through every equation.
The core is reached only through ``core.pipeline`` (plus
``verifier.check_string`` and ``solver.solve_all_orders``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run ui/app.py` puts only the ui/ folder on sys.path, so the
# top-level `core` and `ui` packages would not be importable. Add the project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import streamlit as st  # noqa: E402

from core.pipeline import EXAMPLES_DIR, PipelineResult, list_examples, run_pipeline  # noqa: E402
from core.solver import StepKind, solve_all_orders  # noqa: E402
from core.verifier import check_string  # noqa: E402
from export.report import render_latex, render_markdown, slugify  # noqa: E402
from ui.components import (  # noqa: E402
    build_diagram,
    build_manual_spec,
    editor_rows,
    equations_latex,
    md_escape,
    parse_json_text,
    render_error,
    render_verification,
    show_string,
    step_badge,
)

MAX_COMPARE_STATES = 4
COMPARE_VERIFY_LENGTH = 8

SCHEMA_EXAMPLE = """{
  "name": "strings ending in 01",
  "states": ["q0", "q1", "q2"],
  "alphabet": ["0", "1"],
  "start": "q0",
  "finals": ["q2"],
  "transitions": [
    {"from": "q0", "symbol": "0", "to": "q1"},
    {"from": "q1", "symbol": "1", "to": "q2"}
  ]
}"""


# ---------------------------------------------------------------------------
# Cached core calls (keyed on canonical JSON so equal automata share results)
# ---------------------------------------------------------------------------


def canonical_json(spec: dict[str, Any]) -> str:
    """Stable JSON text for a spec: the cache key."""
    return json.dumps(spec, sort_keys=True, ensure_ascii=False)


@st.cache_data(show_spinner="Solving…", max_entries=64)
def cached_pipeline(spec_json: str, order: tuple[str, ...] | None, verify_length: int) -> PipelineResult:
    """run_pipeline on canonical JSON text (cached)."""
    return run_pipeline(json.loads(spec_json), order=order, verify_length=verify_length)


@st.cache_data(show_spinner="Comparing elimination orders…", max_entries=32)
def cached_compare(spec_json: str, verify_length: int) -> list[dict[str, Any]]:
    """Every elimination order's RE, its length and whether it verifies (cached)."""
    base = cached_pipeline(spec_json, None, verify_length)
    rows = []
    for solved in solve_all_orders(base.system):
        check = cached_pipeline(spec_json, solved.order, verify_length)
        text = str(solved.regex)
        rows.append(
            {
                "order": " → ".join(solved.order),
                "regular expression": text,
                "length": len(text),
                "verified": "yes" if check.verification.passed else "NO",
            }
        )
    rows.sort(key=lambda r: (r["length"], r["regular expression"]))
    best = rows[0]["length"] if rows else None
    for r in rows:
        r["shortest"] = "★" if r["length"] == best else ""
    return rows


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_state() -> None:
    """Default session values."""
    for key, value in {
        "result": None,
        "error": None,
        "step": 0,
        "playing": False,
        "manual_spec": None,
        "spec_json": None,
        "_advance": False,
    }.items():
        st.session_state.setdefault(key, value)


def _set_step(value: int) -> None:
    """Button callback: jump to a step (clamped to the trace)."""
    result: PipelineResult | None = st.session_state.result
    if result is None:
        return
    last = len(result.solve.steps) - 1
    st.session_state.step = max(0, min(last, value))


def _apply_autoplay() -> None:
    """Advance one step if Play scheduled it (before step widgets are created)."""
    result: PipelineResult | None = st.session_state.result
    if not st.session_state.pop("_advance", False) or result is None:
        return
    last = len(result.solve.steps) - 1
    st.session_state.step = min(st.session_state.step + 1, last)
    if st.session_state.step >= last:
        st.session_state.playing = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _example_source() -> dict[str, Any] | None:
    """Example picker."""
    examples = list_examples()
    names = {f: n for f, n, _ in examples}
    descriptions = {f: d for f, _, d in examples}
    default = [f for f, _, _ in examples].index("ends_with_01.json") if "ends_with_01.json" in names else 0
    filename = st.selectbox(
        "Example", list(names), index=default, format_func=lambda f: names[f], key="example_file"
    )
    st.caption(descriptions[filename])
    return parse_json_text((EXAMPLES_DIR / filename).read_text(encoding="utf-8"))


def _upload_source() -> dict[str, Any] | None:
    """JSON upload."""
    with st.expander("Expected JSON format"):
        st.code(SCHEMA_EXAMPLE, language="json")
        st.caption('Use "ε" or "eps" as the symbol of an epsilon transition.')
    uploaded = st.file_uploader("Automaton JSON", type="json", key="upload")
    if uploaded is None:
        st.info("Upload a .json file to continue.")
        return None
    try:
        text = uploaded.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.error("The file is not UTF-8 text. Save it as UTF-8 JSON and upload it again.")
        return None
    try:
        return parse_json_text(text)
    except Exception as exc:  # noqa: BLE001 - shown to the user, never raised
        render_error(exc)
        return None


def _manual_source() -> dict[str, Any] | None:
    """Manual builder: states, alphabet, start, finals and a transition table."""
    states_text = st.text_input("States (comma-separated)", "q0, q1", key="m_states")
    alphabet_text = st.text_input("Alphabet (comma-separated)", "a, b", key="m_alphabet")
    states = [s.strip() for s in states_text.split(",") if s.strip()]
    alphabet = [s.strip() for s in alphabet_text.split(",") if s.strip()]
    start = st.selectbox("Start state", states, key="m_start") if states else None
    finals = st.multiselect("Final states", states, key="m_finals")
    default_rows = {"from": [states[0]] if states else [""],
                    "symbol": [alphabet[0]] if alphabet else [""],
                    "to": [states[-1]] if states else [""]}
    edited = st.data_editor(
        default_rows,
        num_rows="dynamic",
        key="m_transitions",
        column_config={
            "from": st.column_config.SelectboxColumn("from", options=states),
            "symbol": st.column_config.SelectboxColumn("symbol", options=[*alphabet, "ε"]),
            "to": st.column_config.SelectboxColumn("to", options=states),
        },
        width="stretch",
    )
    if st.button("Load into simulator", key="m_load"):
        st.session_state.manual_spec = build_manual_spec(
            states_text, alphabet_text, start, finals, editor_rows(edited)
        )
        st.success("Loaded. Now press Run simulation.")
    if st.session_state.manual_spec is None:
        st.info("Fill in the automaton, then press Load into simulator.")
    return st.session_state.manual_spec


def sidebar() -> None:
    """All inputs; runs the pipeline when Run is pressed."""
    with st.sidebar:
        st.header("1. Automaton")
        mode = st.radio(
            "Input mode", ["Example", "Upload JSON", "Build manually"], key="input_mode",
            horizontal=True,
        )
        if mode == "Example":
            spec = _example_source()
        elif mode == "Upload JSON":
            spec = _upload_source()
        else:
            spec = _manual_source()

        st.header("2. Elimination order")
        order_mode = st.radio("Order", ["Default", "Custom"], key="order_mode", horizontal=True)
        order: tuple[str, ...] | None = None
        order_ok = True
        states = spec.get("states") if isinstance(spec, dict) else None
        if order_mode == "Custom":
            if isinstance(states, list) and all(isinstance(s, str) for s in states):
                picked = st.multiselect(
                    "Pick states in elimination order", states, key="custom_order",
                    help="The first state you pick is eliminated first.",
                )
                if len(picked) < len(states):
                    order_ok = False
                    st.warning(
                        f"Select every state to define a custom order "
                        f"({len(picked)} of {len(states)} chosen)."
                    )
                else:
                    order = tuple(picked)
            else:
                order_ok = False
                st.warning("Load a valid automaton first to choose a custom order.")
        else:
            st.caption("Default: reverse declaration order, start state solved last.")

        st.header("3. Verification")
        verify_length = st.slider("Check all strings up to length", 4, 12, 8, key="verify_length")

        if st.button("Run simulation", type="primary", key="run_btn",
                     disabled=spec is None or not order_ok, width="stretch"):
            st.session_state.step = 0
            st.session_state.playing = False
            st.session_state.error = None
            st.session_state.result = None
            st.session_state.spec_json = canonical_json(spec)  # type: ignore[arg-type]
            try:
                st.session_state.result = cached_pipeline(
                    st.session_state.spec_json, order, verify_length
                )
            except Exception as exc:  # noqa: BLE001 - every error is shown via render_error
                st.session_state.error = exc


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------


def header() -> None:
    """Title, description and the theorem explainer."""
    st.title("Arden's Theorem Simulator")
    st.write("Convert a finite automaton into a regular expression, one equation at a time.")
    with st.expander("What is Arden's Theorem?"):
        st.write("If P and Q are regular expressions and P does not contain the empty string ε, then")
        st.latex(r"R = Q + RP \;\Rightarrow\; R = QP^{*} \qquad \text{when } \varepsilon \notin P")
        st.markdown(
            "**The method, in three steps**\n\n"
            "1. **Write one equation per state.** A state's equation adds up every way to "
            "enter it: for a transition from qⱼ on symbol a, add the term qⱼ·a. The start "
            "state also gets ε (you are there before reading anything).\n"
            "2. **Solve by substitution.** Replace states by their equations one at a time. "
            "Whenever a state refers to itself, q = Q + q·P, Arden's theorem gives q = Q·P\\*.\n"
            "3. **Combine.** The regular expression is the sum (+) of the solutions of all "
            "final states; with no final states it is ∅."
        )


def welcome() -> None:
    """Shown before the first run."""
    st.info(
        "**How to start:** choose an example (or upload / build an automaton) in the "
        "sidebar, optionally pick an elimination order, then press **Run simulation**. "
        "You can then step through every equation the solver writes."
    )


def tab_steps(result: PipelineResult) -> None:
    """Step-by-step walkthrough of the solver trace."""
    steps = result.solve.steps
    last = len(steps) - 1
    st.session_state.step = max(0, min(st.session_state.step, last))

    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1.4, 1.6])
    c1.button("⏮ First", key="btn_first", on_click=_set_step, args=(0,), width="stretch")
    c2.button("◀ Prev", key="btn_prev", on_click=_set_step,
              args=(st.session_state.step - 1,), width="stretch")
    c3.button("Next ▶", key="btn_next", on_click=_set_step,
              args=(st.session_state.step + 1,), width="stretch")
    c4.button("Last ⏭", key="btn_last", on_click=_set_step, args=(last,), width="stretch")
    c5.toggle("▶ Play", key="playing", help="Advance automatically about once per second.")
    c6.markdown(f"**Step {st.session_state.step + 1} of {len(steps)}**")
    st.slider("Step", 0, last, key="step", label_visibility="collapsed")

    step = steps[st.session_state.step]
    left, right = st.columns([1, 1.25])
    with left:
        st.graphviz_chart(
            build_diagram(result.normalized, step.focus_state, step.affected_states),
            width="stretch",
        )
        if result.normalized != result.original:
            st.caption("Diagram shows the automaton after preprocessing (see the Preprocessing tab).")
    with right:
        label, color = step_badge(step)
        st.markdown(f":{color}-background[{label}]")
        st.markdown(md_escape(step.explanation))
        if step.rule_latex:
            st.latex(step.rule_latex)
        if step.kind is StepKind.RESULT:
            st.success("Final regular expression")
            st.latex(r"\Large " + result.regex_latex)
        else:
            st.caption("Before")
            st.latex(equations_latex(step.system_before, step.affected_states))
            st.caption("After")
            st.latex(equations_latex(step.system_after, step.affected_states))


def tab_preprocessing(result: PipelineResult) -> None:
    """Normalization steps with before/after diagrams."""
    for i, pre in enumerate(result.preprocessing, start=1):
        if pre.kind == "none":
            st.info(pre.description)
            continue
        st.subheader(f"{i}. {pre.kind.replace('_', ' ').capitalize()}")
        st.markdown(md_escape(pre.description))
        before, after = st.columns(2)
        with before:
            st.caption("Before")
            st.graphviz_chart(build_diagram(pre.before, affected=pre.affected_states),
                              width="stretch")
        with after:
            st.caption("After")
            st.graphviz_chart(build_diagram(pre.after), width="stretch")


def tab_result(result: PipelineResult) -> None:
    """Final RE, closed forms, verification and the string tester."""
    for note in result.notes:
        st.info(note)
    st.subheader("Regular expression")
    st.code(result.regex_text, language=None)
    st.latex(result.regex_latex)

    st.subheader("Closed form of every state")
    finals = result.normalized.finals
    st.dataframe(
        [
            {"state": s, "final": "yes" if s in finals else "", "closed form": str(r)}
            for s, r in result.solve.closed_forms.items()
        ],
        hide_index=True,
        width="stretch",
    )

    st.subheader("Verification")
    v = result.verification
    render_verification(v)
    acc, rej = st.columns(2)
    with acc:
        st.caption("Accepted by both (shortest first)")
        st.code("\n".join(show_string(s) for s in v.accepted_samples) or "(none)", language=None)
    with rej:
        st.caption("Rejected by both (shortest first)")
        st.code("\n".join(show_string(s) for s in v.rejected_samples) or "(none)", language=None)

    st.subheader("Try a string")
    word = st.text_input("String to test (leave empty to test ε)", key="try_string")
    fa, rx = check_string(result.original, result.solve.regex, word)
    shown = md_escape(show_string(word))
    left, right = st.columns(2)
    (left.success if fa else left.error)(f"Automaton: {'accepts' if fa else 'rejects'} **{shown}**")
    (right.success if rx else right.error)(f"Regular expression: {'matches' if rx else 'does not match'} **{shown}**")
    if fa != rx:
        st.warning("The automaton and the regular expression disagree on this string!")


def tab_compare(result: PipelineResult, spec_json: str) -> None:
    """All elimination orders side by side (small automata only)."""
    n = len(result.normalized.states)
    if n > MAX_COMPARE_STATES:
        st.info(
            f"The automaton has {n} states after preprocessing, i.e. {math.factorial(n):,} "
            "possible orders. Comparing "
            f"every order is limited to {MAX_COMPARE_STATES} states to keep this fast."
        )
        return
    st.write(
        "Different elimination orders substitute the equations in a different sequence, so "
        "the regular expressions look different, but each one solves the same equation "
        "system, so they all describe the same language."
    )
    length = min(st.session_state.get("verify_length", 8), COMPARE_VERIFY_LENGTH)
    rows = cached_compare(spec_json, length)
    st.dataframe(
        rows, hide_index=True, width="stretch",
        column_order=["shortest", "order", "regular expression", "length", "verified"],
    )
    st.caption(f"Sorted by length (characters); ★ marks the shortest. Verified up to length {length}.")


def tab_export(result: PipelineResult) -> None:
    """Download the report (Markdown / LaTeX) and the automaton JSON."""
    slug = slugify(result.original.name)
    st.write("Download a full report with every step of the conversion.")
    md_col, tex_col = st.columns(2)
    md_col.download_button(
        "Download Markdown report", data=render_markdown(result),
        file_name=f"{slug}.md", mime="text/markdown", key="dl_md", width="stretch",
    )
    tex_col.download_button(
        "Download LaTeX report", data=render_latex(result),
        file_name=f"{slug}.tex", mime="application/x-tex", key="dl_tex", width="stretch",
    )
    st.caption("The .tex file is a complete document: it compiles in Overleaf or with `pdflatex`.")
    st.divider()
    st.write("Save the automaton itself, e.g. one you built manually, to upload it again later.")
    st.download_button(
        "Download automaton JSON",
        data=json.dumps(result.original.to_dict(), indent=2, ensure_ascii=False),
        file_name=f"{slug}.json", mime="application/json", key="dl_json",
    )


def main() -> None:
    """Build the page."""
    st.set_page_config(page_title="Arden's Theorem Simulator", layout="wide")
    _init_state()
    sidebar()
    header()
    _apply_autoplay()

    if st.session_state.error is not None:
        render_error(st.session_state.error)
    result: PipelineResult | None = st.session_state.result
    if result is None:
        if st.session_state.error is None:
            welcome()
        return

    st.caption(f"Showing: **{md_escape(result.original.name)}**")
    spec_json = st.session_state.spec_json or canonical_json(result.original.to_dict())
    tabs = st.tabs(["Step-by-step", "Preprocessing", "Result & verification", "Compare orders", "Export"])
    for tab, render in zip(tabs, (tab_steps, tab_preprocessing, tab_result)):
        with tab:
            try:
                render(result)
            except Exception as exc:  # noqa: BLE001
                render_error(exc)
    with tabs[3]:
        try:
            tab_compare(result, spec_json)
        except Exception as exc:  # noqa: BLE001
            render_error(exc)
    with tabs[4]:
        try:
            tab_export(result)
        except Exception as exc:  # noqa: BLE001
            render_error(exc)

    if st.session_state.playing and st.session_state.step < len(result.solve.steps) - 1:
        time.sleep(1.0)
        st.session_state._advance = True
        st.rerun()


main()
