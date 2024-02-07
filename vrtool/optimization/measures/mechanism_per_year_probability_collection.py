from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear


class MechanismPerYearProbabilityCollection:
    _probabilities: list[MechanismPerYear]

    def __init__(self, probabilities: list[MechanismPerYear]) -> None:
        self._probabilities = probabilities

    def filter(self, mechanism: MechanismEnum, year: int) -> float:
        for p in self._probabilities:
            if p.mechanism == mechanism and p.year == year:
                return p.probability
        raise ValueError("mechanism/year not found")
