from __future__ import annotations

from vrtool.optimization.models.mechanism_per_year import MechanismPerYear
from vrtool.common.enums.mechanism_enum import MechanismEnum

class MechanismPerYearProbabilityCollection:
    _probabilities : list[MechanismPerYear]

    def __init__(self, probabilities: list[MechanismPerYear]) -> None:
        self._probabilities = probabilities

    def filter(self, mechanism: MechanismEnum, year: int) -> float:
        for p in self._probabilities:
            if p.mechanism == mechanism and p.year == year:
                return p.probability
        raise ValueError("mechanism/year not found")
