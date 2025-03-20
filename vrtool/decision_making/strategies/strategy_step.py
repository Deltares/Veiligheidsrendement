from dataclasses import dataclass, field

import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.enums.step_type_enum import StepTypeEnum
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)


@dataclass(kw_only=True)
class StrategyStep:
    step_type: StepTypeEnum = StepTypeEnum.SINGLE
    measure_taken: tuple[int, int, int] = field(default_factory=lambda: (0,0,0))
    cost: float = 0.0
    risk: float = 0.0
    probabilities: dict[MechanismEnum, np.ndarray] = field(default_factory=dict)
    selected_aggregated_measure: tuple[int, AggregatedMeasureCombination] = field(default_factory=tuple)
