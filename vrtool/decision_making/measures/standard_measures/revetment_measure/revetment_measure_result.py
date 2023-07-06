from dataclasses import dataclass
from typing import Union

from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_data import (
    RevetmentMeasureData,
)


@dataclass
class RevetmentMeasureResult:
    year: int
    beta_target: float
    beta_combined: float
    transition_level: float
    cost: float
    revetment_measures: Union[
        list[RevetmentMeasureData], list[list[RevetmentMeasureData]]
    ]
