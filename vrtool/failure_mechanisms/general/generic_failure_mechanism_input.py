from __future__ import annotations

from dataclasses import dataclass

from vrtool.flood_defence_system.mechanism_input import MechanismInput


@dataclass
class GenericFailureMechanismInput:
    time_grid: list[int]
    beta_grid: list[float]

    @classmethod
    def from_mechanism_input(
        cls, mechanism_input: MechanismInput
    ) -> GenericFailureMechanismInput:
        _time_grid = list(mechanism_input.input["beta"].keys())
        _beta_grid = list(mechanism_input.input["beta"].values())

        return cls(time_grid=_time_grid, beta_grid=_beta_grid)
