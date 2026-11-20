"""Exact, standard-library-only prototype of integer recurrent spiking modules."""

from .model import (
    Params,
    State,
    clip,
    iter_legal_states,
    legal_state_count,
    step,
    trunc_div,
    validate_state,
    zero_state,
)
from .reference import reference_step

__all__ = [
    "Params",
    "State",
    "clip",
    "iter_legal_states",
    "legal_state_count",
    "reference_step",
    "step",
    "trunc_div",
    "validate_state",
    "zero_state",
]
