from dataclasses import dataclass, field

import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.enums.step_type_enum import StepTypeEnum
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)


@dataclass(kw_only=True)
class StrategyStep:
    step_number: int = 0
    step_type: StepTypeEnum = StepTypeEnum.UNKNOWN
    measure: tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))
    section_idx: int = 0
    aggregated_measure: AggregatedMeasureCombination = None
    probabilities: dict[MechanismEnum, np.ndarray] = field(default_factory=dict)
    bc_ratio: float = 0.0
    total_risk: float = 0.0
    total_cost: float = 0.0
