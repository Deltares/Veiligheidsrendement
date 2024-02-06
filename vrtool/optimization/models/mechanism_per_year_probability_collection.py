from __future__ import annotations

from vrtool.common.enums.mechanism_enum import MechanismEnum

class MechanismPerYearProbabilityCollection:
    _probabilities : {{}}

    def __init__(self, probabilities: dict) -> None:
        self._probabilities = probabilities

    def filter(self, mechanism: MechanismEnum, year: int) -> float:
        return self._probabilities[mechanism][year];
