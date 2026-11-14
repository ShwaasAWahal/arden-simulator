"""Tests for core.pipeline: facade, order adjustment, errors, CLI."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

import core.pipeline as pipeline_mod
from core import ArdenError, ArdenPreconditionError, InvalidAutomatonError, SolverError
from core.automaton import load_automaton
from core.pipeline import format_result, list_examples, main, run_pipeline
from core.regex_ast import concat, star, sym, to_latex, to_python_re, to_text, union
from core.solver import default_order
from core.verifier import verify

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
EXAMPLE_FILES = sorted(p.name for p in EXAMPLES_DIR.glob("*.json"))


def raw(name: str) -> dict:
    return json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# run_pipeline on every example and every input kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
@pytest.mark.parametrize("kind", ["dict", "json", "path", "str_path", "automaton"])
def test_every_example_every_input_kind(filename: str, kind: str) -> None:
    path = EXAMPLES_DIR / filename
    source = {
        "dict": lambda: raw(filename),
        "json": lambda: path.read_text(encoding="utf-8"),
        "path": lambda: path,
        "str_path": lambda: str(path),
        "automaton": lambda: load_automaton(path),
    }[kind]()
    result = run_pipeline(source)
    assert result.verification.passed, (filename, result.verification.counterexample)
    assert result.original == load_automaton(path)
    assert result.notes == ()
    assert result.regex_text == to_text(result.solve.regex)
    assert result.regex_latex == to_latex(result.solve.regex)
    assert result.regex_python == to_python_re(result.solve.regex)
    assert result.system == result.solve.steps[0].system_before
    assert result.preprocessing


def test_pipeline_stages_are_wired() -> None:
    result = run_pipeline(EXAMPLES_DIR / "epsilon_nfa.json", verify_length=5)
    assert result.original.has_epsilon() and not result.normalized.has_epsilon()
    assert result.preprocessing[0].kind == "remove_epsilon"
    assert result.system.state_order == result.normalized.states
    assert result.verification.max_length == 5


def test_list_examples() -> None:
    examples = list_examples()
    assert [f for f, _, _ in examples] == EXAMPLE_FILES
    assert all(name and desc for _, name, desc in examples)
    assert ("ends_with_01.json", "strings ending in 01",
            "Binary strings whose last two symbols are 0 then 1.") in examples


# ---------------------------------------------------------------------------
# Custom orders
# ---------------------------------------------------------------------------


def test_custom_order_used() -> None:
    result = run_pipeline(EXAMPLES_DIR / "ends_with_01.json", order=["q0", "q1", "q2"])
    assert result.solve.order == ("q0", "q1", "q2")
    assert result.notes == ()
    assert result.verification.passed


def test_order_with_removed_states_is_adjusted() -> None:
    # unreachable_dead: q3 is unreachable and q2 is dead; both vanish in normalize().
    result = run_pipeline(EXAMPLES_DIR / "unreachable_dead.json", order=["q3", "q2", "q1", "q0"])
    assert result.normalized.states == ("q0", "q1")
    assert result.solve.order == ("q1", "q0")
    assert len(result.notes) == 1
    assert "q3, q2" in result.notes[0] and "removed during normalization" in result.notes[0]
    assert result.verification.passed


def test_order_missing_states_are_appended_in_default_order() -> None:
    result = run_pipeline(EXAMPLES_DIR / "unreachable_dead.json", order=["q0", "q3"])
    assert result.solve.order == ("q0", "q1")
    assert any("q3" in n and "dropped" in n for n in result.notes)
    assert any("q1" in n and "appended" in n for n in result.notes)
    clean = run_pipeline(EXAMPLES_DIR / "ends_with_01.json", order=["q0"])
    tail = [s for s in default_order(clean.system) if s != "q0"]
    assert clean.solve.order == ("q0", *tail)
    assert result.verification.passed and clean.verification.passed


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_bad_json_text() -> None:
    with pytest.raises(InvalidAutomatonError, match="Invalid JSON"):
        run_pipeline('{"states": [')


def test_invalid_automaton() -> None:
    data = raw("ends_with_01.json")
    data["start"] = "q9"
    with pytest.raises(InvalidAutomatonError, match="start state 'q9'"):
        run_pipeline(data)


def test_missing_file() -> None:
    with pytest.raises(InvalidAutomatonError, match="Cannot read"):
        run_pipeline(EXAMPLES_DIR / "does_not_exist.json")
    with pytest.raises(InvalidAutomatonError, match="Cannot read"):
        run_pipeline("no/such/file.json")


def test_unsupported_source_type() -> None:
    with pytest.raises(InvalidAutomatonError, match="Cannot read an automaton from a int"):
        run_pipeline(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("order", "fragment"),
    [
        (["q2", "q1", "qq"], "mentions qq, which is not a state"),
        (["x", "y"], "mentions x, y, which are not states"),
        (["q2", "q2", "q1", "q0"], "lists q2 more than once"),
    ],
)
def test_bad_order(order: list[str], fragment: str) -> None:
    with pytest.raises(SolverError, match=fragment):
        run_pipeline(EXAMPLES_DIR / "ends_with_01.json", order=order)


def test_negative_verify_length() -> None:
    with pytest.raises(ArdenError, match="0 or more"):
        run_pipeline(EXAMPLES_DIR / "ends_with_01.json", verify_length=-1)


def test_arden_precondition_error_class_exists_in_hierarchy() -> None:
    # Cannot be triggered through run_pipeline (normalize removes ε-moves), but
    # the facade documents it; make sure it is still catchable as ArdenError.
    assert issubclass(ArdenPreconditionError, ArdenError)


# ---------------------------------------------------------------------------
# Report formatting and CLI
# ---------------------------------------------------------------------------


def test_format_result_with_trace() -> None:
    result = run_pipeline(EXAMPLES_DIR / "unreachable_dead.json", order=["q3", "q0", "q1"])
    text = format_result(result, trace=True)
    for fragment in (
        "== Preprocessing ==", "Removed unreachable state(s) q3", "== Initial equations ==",
        "== Trace ==", "[0] INITIAL", "ARDEN", "Note: State(s) q3",
        f"RE:    {result.regex_text}", "PASSED",
    ):
        assert fragment in text, fragment
    assert "== Trace ==" not in format_result(result)


def test_format_result_failed_verification() -> None:
    result = run_pipeline(EXAMPLES_DIR / "ends_with_01.json")
    wrong = verify(result.original, concat(star(union(sym("0"), sym("1"))), sym("0")))
    text = format_result(dataclasses.replace(result, verification=wrong))
    assert 'FAILED: on "0" the automaton rejects but the RE accepts' in text
    empty = verify(result.original, union(sym("1"), concat()))  # wrong on ""
    assert 'on "ε"' in format_result(dataclasses.replace(result, verification=empty))


def test_format_result_truncated_note() -> None:
    result = run_pipeline(EXAMPLES_DIR / "epsilon_nfa.json", verify_length=20)
    assert result.verification.truncated
    assert "reduced to stay within the string budget" in format_result(result)


def test_main_in_process(capsys: pytest.CaptureFixture[str]) -> None:
    path = str(EXAMPLES_DIR / "ends_with_01.json")
    assert main([path, "--order", "q0, q1,q2", "--verify-length", "6", "--trace"]) == 0
    out = capsys.readouterr().out
    assert "Elimination order: q0, q1, q2" in out and "[0] INITIAL" in out


def test_main_exit_codes(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main([str(bad)]) == 2
    assert "Error: Invalid JSON" in capsys.readouterr().err
    assert main([str(EXAMPLES_DIR / "ends_with_01.json"), "--order", "zz"]) == 2
    assert "not a state" in capsys.readouterr().err

    real = pipeline_mod.run_pipeline

    def failing(*args: object, **kwargs: object):
        r = real(*args, **kwargs)
        wrong = verify(r.original, sym("1"))
        return dataclasses.replace(r, verification=wrong)

    monkeypatch.setattr(pipeline_mod, "run_pipeline", failing)
    assert main([str(EXAMPLES_DIR / "ends_with_01.json")]) == 1
    assert "FAILED" in capsys.readouterr().out


def _cli(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-m", "core.pipeline", *args],
        cwd=ROOT,
        capture_output=True,
        timeout=120,
    )


def test_cli_subprocess_success() -> None:
    proc = _cli("examples/ends_with_01.json")
    out = proc.stdout.decode("utf-8")
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    expected = run_pipeline(EXAMPLES_DIR / "ends_with_01.json").regex_text
    assert f"RE:    {expected}" in out
    assert "PASSED" in out


def test_cli_subprocess_malformed_file(tmp_path: Path) -> None:
    bad = tmp_path / "malformed.json"
    bad.write_text('{"states": ["q0"], "alphabet": [', encoding="utf-8")
    proc = _cli(str(bad))
    assert proc.returncode == 2
    assert "Invalid JSON" in proc.stderr.decode("utf-8")
    assert proc.stdout == b""
