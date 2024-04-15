from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SectionYearProbabilities:
    """
    Class to store the probabilities per year for a section.
    """

    year_probabilities: list[float] = field(default_factory=list)

    @classmethod
    def from_strategy_input(cls, probabilities: np.ndarray) -> SectionYearProbabilities:
        _section_year_prob = cls()
        _section_year_prob.year_probabilities = probabilities.tolist()
        return _section_year_prob

    def get_probability(self) -> float:
        return max(self.year_probabilities)
