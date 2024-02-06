from __future__ import annotations

class MechanismPerYearProbabilityCollection:
    _probabilities : {{}}

    def __init__(self, probabilities: dict) -> None:
        self._probabilities = probabilities

    def filter(self, mechanism: int, year: int) -> float:
        return self._probabilities[mechanism][year];
