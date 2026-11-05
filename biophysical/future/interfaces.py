"""Interfaces for possible future modules. NOTHING here is implemented or validated.

These protocols record what a component must declare before it may be composed with others. They
exist so future work has an explicit contract; importing them adds no scientific capability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ModelAssumptions:
    """Required by any mechanistic or learned dynamics module."""
    biological_problem: str
    governing_equations: str
    parameters_with_units: dict[str, str]
    boundary_conditions: str
    assumptions: tuple[str, ...]
    validation_criteria: tuple[str, ...]
    conventional_baseline: str
    simulated_output: bool = True   # any trajectory produced is simulation, and must be labelled so


@dataclass
class Trajectory:
    t: np.ndarray
    states: np.ndarray
    state_names: list[str]
    units: dict[str, str]
    is_simulated: bool = True
    provenance: dict = field(default_factory=dict)


@runtime_checkable
class MechanisticModel(Protocol):
    """ODE, stochastic kinetics or reaction-diffusion model of a defined pathway."""
    assumptions: ModelAssumptions

    def simulate(self, initial_state: np.ndarray, t: np.ndarray, parameters: dict[str, float]) -> Trajectory: ...


@runtime_checkable
class TransitionOperator(Protocol):
    """Learned state-to-state map; requires time-resolved training data this project does not have."""
    def step(self, state: np.ndarray, dt: float, perturbation: tuple[str, ...]) -> np.ndarray: ...
