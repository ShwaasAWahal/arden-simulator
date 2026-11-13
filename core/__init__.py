"""Core of the Arden simulator: pure, testable FA -> RE conversion logic.

This package must never import streamlit, graphviz, or anything from ``ui/``
or ``export/``. It also defines the exception hierarchy used for all errors
caused by user input.
"""

from __future__ import annotations

__all__ = [
    "ArdenError",
    "InvalidAutomatonError",
    "ArdenPreconditionError",
    "SolverError",
]


class ArdenError(Exception):
    """Base class for every error raised by the Arden simulator core."""


class InvalidAutomatonError(ArdenError):
    """The input automaton is malformed or violates the JSON schema/conventions.

    Examples: unknown states in transitions, a symbol outside the alphabet,
    a missing or duplicate start state, or a reserved alphabet character.
    """


class ArdenPreconditionError(ArdenError):
    """Arden's theorem was applied where its precondition does not hold.

    Arden's theorem (R = Q + RP  =>  R = QP*) requires that P is not
    nullable (P does not contain epsilon); otherwise the solution is not unique.
    """


class SolverError(ArdenError):
    """The equation system could not be solved (e.g. an invalid elimination order)."""
