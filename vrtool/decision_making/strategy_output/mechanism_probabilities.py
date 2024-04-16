from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategy_output.section_year_probabilities import (
    SectionYearProbabilities,
)


@dataclass
class MechanismProbabilities:
    """
    Class to store the probabilities of a mechanism in a specific year.
    """

    mechanism: MechanismEnum = None
    section_probabilities: list[SectionYearProbabilities] = field(default_factory=list)

    @classmethod
    def from_strategy_input(
        cls,
        mechanism: MechanismEnum,
        mechanism_prob: np.ndarray,
        sh_idx: int,
        sg_idx: int,
    ) -> MechanismProbabilities:
        _mechanism_prob = cls(mechanism=mechanism)
        # Filter out the probabilities based on the given measure index
        _idx = sg_idx
        if mechanism in (MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT):
            _idx = sh_idx
        _mechanism_prob.section_probabilities = list(
            map(
                SectionYearProbabilities.from_strategy_input,
                (_section_prob[_idx, :] for _section_prob in mechanism_prob),
            )
        )
        return _mechanism_prob

    def get_probabilities(self) -> np.ndarray:
        if self.mechanism in (MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT):
            return np.max(
                list(map(lambda x: x.get_probabilities(), self.section_probabilities)),
                axis=0,
            )
        if self.mechanism in (MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING):
            return np.sum(
                list(map(lambda x: x.get_probabilities(), self.section_probabilities)),
                axis=0,
            )
        return np.array([])
