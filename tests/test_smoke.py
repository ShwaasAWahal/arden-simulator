"""Smoke tests: every core module imports and the exception hierarchy is wired up."""

from __future__ import annotations

import importlib

import pytest

import core
from core import (
    ArdenError,
    ArdenPreconditionError,
    InvalidAutomatonError,
    SolverError,
)

CORE_MODULES = [
    "core.automaton",
    "core.regex_ast",
    "core.equations",
    "core.solver",
    "core.verifier",
    "core.pipeline",
]


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_module_imports(name: str) -> None:
    """Each core module imports cleanly and has a module docstring."""
    module = importlib.import_module(name)
    assert module.__doc__ and module.__doc__.strip()


@pytest.mark.parametrize(
    "exc", [InvalidAutomatonError, ArdenPreconditionError, SolverError]
)
def test_exceptions_derive_from_arden_error(exc: type[Exception]) -> None:
    """All user-facing errors are catchable as core.ArdenError."""
    assert issubclass(exc, ArdenError)
    with pytest.raises(ArdenError):
        raise exc("boom")


def test_arden_error_is_exception() -> None:
    """The root of the hierarchy is a plain Exception subclass."""
    assert issubclass(ArdenError, Exception)
    assert set(core.__all__) >= {
        "ArdenError",
        "InvalidAutomatonError",
        "ArdenPreconditionError",
        "SolverError",
    }
