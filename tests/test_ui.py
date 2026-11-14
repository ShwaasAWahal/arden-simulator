"""Tests for the Streamlit UI: AppTest smoke/flow tests and pure component tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import InvalidAutomatonError
from core.pipeline import run_pipeline
from ui.components import (
    AFFECTED_COLOR,
    FOCUS_COLOR,
    build_diagram,
    build_manual_spec,
    editor_rows,
    equations_latex,
    md_escape,
    parse_json_text,
    show_string,
    step_badge,
)

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "ui" / "app.py")
EXAMPLES_DIR = ROOT / "examples"


# ---------------------------------------------------------------------------
# AppTest helpers
# ---------------------------------------------------------------------------


def fresh_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def run_example(at: AppTest, filename: str = "ends_with_01.json") -> AppTest:
    at.selectbox(key="example_file").set_value(filename).run()
    at.button(key="run_btn").click().run()
    assert not at.exception, at.exception
    return at


def markdown_texts(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown]


def step_indicator(at: AppTest) -> str:
    return next(t for t in markdown_texts(at) if t.startswith("**Step "))


# ---------------------------------------------------------------------------
# App flows
# ---------------------------------------------------------------------------


def test_app_loads_with_welcome() -> None:
    at = fresh_app()
    assert at.title[0].value == "Arden's Theorem Simulator"
    assert any("How to start" in i.value for i in at.info)
    assert not at.tabs  # no results yet


def test_run_example_shows_final_regex() -> None:
    at = run_example(fresh_app())
    expected = run_pipeline(EXAMPLES_DIR / "ends_with_01.json").regex_text
    assert expected in [c.value for c in at.code]
    assert any("Verified" in s.value for s in at.success)
    assert len(at.tabs) == 5


def test_step_navigation() -> None:
    at = run_example(fresh_app())
    n = len(run_pipeline(EXAMPLES_DIR / "ends_with_01.json").solve.steps)
    assert step_indicator(at) == f"**Step 1 of {n}**"
    at.button(key="btn_next").click().run()
    assert step_indicator(at) == f"**Step 2 of {n}**"
    at.button(key="btn_prev").click().run()
    assert step_indicator(at) == f"**Step 1 of {n}**"
    at.button(key="btn_last").click().run()
    assert step_indicator(at) == f"**Step {n} of {n}**"
    assert any(":green-background[Result]" == t for t in markdown_texts(at))
    assert any(s.value == "Final regular expression" for s in at.success)
    at.button(key="btn_first").click().run()
    assert step_indicator(at) == f"**Step 1 of {n}**"
    assert not at.exception


def test_slider_moves_step() -> None:
    at = run_example(fresh_app())
    at.slider(key="step").set_value(3).run()
    assert step_indicator(at).startswith("**Step 4 of")


def test_new_run_resets_step() -> None:
    at = run_example(fresh_app())
    at.button(key="btn_last").click().run()
    run_example(at, "even_zeros.json")
    assert step_indicator(at).startswith("**Step 1 of")


def test_custom_order_missing_state_warns() -> None:
    at = fresh_app()
    at.radio(key="order_mode").set_value("Custom").run()
    at.multiselect(key="custom_order").set_value(["q0"]).run()
    assert not at.exception
    assert any("Select every state" in w.value and "1 of 3" in w.value for w in at.warning)
    assert at.button(key="run_btn").disabled

    at.multiselect(key="custom_order").set_value(["q0", "q1", "q2"]).run()
    assert not any("Select every state" in w.value for w in at.warning)
    at.button(key="run_btn").click().run()
    assert not at.exception
    assert any("Elimination order: q0, q1, q2" in t for t in markdown_texts(at))


def test_try_a_string() -> None:
    at = run_example(fresh_app())
    at.text_input(key="try_string").input("1101").run()
    texts = [s.value for s in at.success]
    assert any("Automaton: accepts **1101**" in t for t in texts)
    assert any("Regular expression: matches **1101**" in t for t in texts)
    at.text_input(key="try_string").input("10").run()
    assert any("Automaton: rejects **10**" in e.value for e in at.error)
    assert not at.warning  # FA and RE never disagree


def test_compare_orders_table() -> None:
    at = run_example(fresh_app())
    table = at.dataframe[-1].value
    assert len(table) == 6  # 3! orders
    assert set(table["verified"]) == {"yes"}
    assert list(table["length"]) == sorted(table["length"])
    assert table["shortest"].iloc[0] == "★"


def test_export_tab_download_buttons() -> None:
    at = run_example(fresh_app())
    buttons = at.get("download_button")
    labels = [b.proto.label for b in buttons]
    assert labels == ["Download Markdown report", "Download LaTeX report", "Download automaton JSON"]
    assert any("pdflatex" in c.value for c in at.caption)
    assert not at.exception
    assert not any("went wrong" in e.value for e in at.error)


def test_epsilon_example_preprocessing_tab() -> None:
    at = run_example(fresh_app(), "epsilon_nfa.json")
    assert any("Removed 2 ε-transition" in t for t in markdown_texts(at))
    assert not at.exception


def test_manual_builder_flow() -> None:
    at = fresh_app()
    at.radio(key="input_mode").set_value("Build manually").run()
    at.multiselect(key="m_finals").set_value(["q1"]).run()
    at.button(key="m_load").click().run()
    at.button(key="run_btn").click().run()
    assert not at.exception
    # default row: q0 --a--> q1, with q1 final  =>  RE "a"
    assert "a" in [c.value for c in at.code]


def test_manual_builder_invalid_input_is_friendly() -> None:
    at = fresh_app()
    at.radio(key="input_mode").set_value("Build manually").run()
    at.text_input(key="m_alphabet").input("a, ab").run()
    at.button(key="m_load").click().run()
    at.button(key="run_btn").click().run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("The automaton is not valid" in e and "not a single character" in e for e in errors)


# ---------------------------------------------------------------------------
# Upload helper
# ---------------------------------------------------------------------------


def test_parse_json_text_malformed() -> None:
    with pytest.raises(InvalidAutomatonError) as info:
        parse_json_text('{"states": ["q0",')
    msg = str(info.value)
    assert "not valid JSON" in msg and "line 1" in msg


def test_parse_json_text_not_object() -> None:
    with pytest.raises(InvalidAutomatonError, match="one JSON object"):
        parse_json_text("[1, 2, 3]")


def test_parse_json_text_ok() -> None:
    text = (EXAMPLES_DIR / "ends_with_01.json").read_text(encoding="utf-8")
    assert parse_json_text(text)["start"] == "q0"


# ---------------------------------------------------------------------------
# Components (no rendering)
# ---------------------------------------------------------------------------

PARALLEL = {
    "name": "parallel",
    "states": ["q0", "q1", "q2"],
    "alphabet": ["0", "1"],
    "start": "q0",
    "finals": ["q1", "q2"],
    "transitions": [
        {"from": "q0", "symbol": "0", "to": "q1"},
        {"from": "q0", "symbol": "1", "to": "q1"},
        {"from": "q1", "symbol": "1", "to": "q2"},
    ],
}


def test_diagram_structure() -> None:
    aut = run_pipeline(PARALLEL).original
    src = build_diagram(aut).source
    assert "rankdir=LR" in src
    assert src.count("doublecircle") == 2  # q1 and q2
    assert 'label="0, 1"' in src  # merged parallel edges
    assert src.count("q0 -> q1") == 1
    assert "__start__ -> q0" in src
    assert "bgcolor=white" in src


def test_diagram_highlights() -> None:
    aut = run_pipeline(PARALLEL).original
    src = build_diagram(aut, focus_state="q1", affected=("q2",)).source
    q1_line = next(l for l in src.splitlines() if l.strip().startswith("q1 [") )
    q2_line = next(l for l in src.splitlines() if l.strip().startswith("q2 ["))
    assert FOCUS_COLOR in q1_line
    assert AFFECTED_COLOR in q2_line
    assert FOCUS_COLOR not in build_diagram(aut).source


def test_diagram_epsilon_label_first() -> None:
    aut = run_pipeline(EXAMPLES_DIR / "epsilon_nfa.json").original
    assert 'label="ε"' in build_diagram(aut).source


def test_equations_latex_highlight() -> None:
    system = run_pipeline(EXAMPLES_DIR / "ends_with_01.json").system
    plain = equations_latex(system)
    assert "\\color" not in plain
    assert plain.startswith("\\begin{aligned}") and "&=" in plain
    marked = equations_latex(system, highlight=("q1",))
    assert marked.count("\\color{") == 2  # lhs and rhs of q1
    q1_line = next(l for l in marked.splitlines() if "q_{1}" in l.split("&=")[0])
    assert "\\color" in q1_line


def test_step_badges() -> None:
    steps = run_pipeline(EXAMPLES_DIR / "ends_with_01.json").solve.steps
    badges = [step_badge(s) for s in steps]
    assert badges[0] == ("Setup · Initial system", "gray")
    assert badges[-1] == ("Result", "green")
    assert ("Elimination · Arden's theorem", "violet") in badges


def test_text_helpers() -> None:
    assert show_string("") == "ε" and show_string("01") == "01"
    assert md_escape("(a+b)*a_b") == r"(a+b)\*a\_b"


def test_manual_spec_helpers() -> None:
    rows = editor_rows({"from": ["q0", None], "symbol": ["a", None], "to": ["q1", None]})
    assert rows == [{"from": "q0", "symbol": "a", "to": "q1"},
                    {"from": None, "symbol": None, "to": None}]
    spec = build_manual_spec("q0, q1", "a,b", "q0", ["q1"], rows)
    assert spec["states"] == ["q0", "q1"] and spec["alphabet"] == ["a", "b"]
    assert spec["transitions"] == [{"from": "q0", "symbol": "a", "to": "q1"}]
    assert run_pipeline(spec).regex_text == "a"
    assert editor_rows([{"from": "q0"}]) == [{"from": "q0"}]


def test_render_error_and_verification() -> None:
    def script() -> None:
        from core import InvalidAutomatonError
        from core.pipeline import run_pipeline
        from core.regex_ast import EPS  # only to build a wrong regex for this test
        from core.verifier import verify
        from ui.components import render_error, render_verification

        render_error(InvalidAutomatonError("start state 'q9' is not in the state list"))
        render_error(RuntimeError("boom"))
        original = run_pipeline("examples/ends_with_01.json").original
        render_verification(verify(original, EPS))

    at = AppTest.from_function(script, default_timeout=60)
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert errors[0].startswith("**The automaton is not valid.**")
    assert "Something went wrong inside the simulator" in errors[1]
    assert at.expander[0].label == "Technical details"
    assert "Mismatch on the string **ε**" in errors[2]
    assert "automaton rejects it but the regular expression accepts it" in errors[2]
