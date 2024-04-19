from typing import Protocol

import numpy as np
from typing_extensions import runtime_checkable

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.section_as_input import SectionAsInput


@runtime_checkable
class StrategyProtocol(Protocol):
    design_method: str
    sections: list[SectionAsInput]
    time_periods: list[int]
    measures_taken: list[tuple[int, int, int]]
    total_risk_per_step: list[float]
    probabilities_per_step: list[dict[MechanismEnum, np.ndarray]]

    def evaluate(self, *args, **kwargs):
        """
        Evaluates the provided measures.
        TODO: For now the arguments are not specific as we do not have a clear view
        on which generic input structures will be used across the different strategies.
        """
