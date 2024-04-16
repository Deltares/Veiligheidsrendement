from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SectionYearProbabilities:
    """
    Class to store the probabilities per year for a section.
    """

    probabilities: np.ndarray = np.array([])

    @classmethod
    def from_strategy_input(cls, probabilities: np.ndarray) -> SectionYearProbabilities:
        return cls(probabilities=probabilities)

    def get_probabilities(self) -> np.ndarray:
        return self.probabilities
