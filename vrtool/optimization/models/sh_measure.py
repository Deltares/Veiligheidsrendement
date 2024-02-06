from __future__ import annotations

from dataclasses import dataclass

from measure_as_input_base import MeasureAsInputBase, MechanismPerYearProbabilityCollection
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum

@dataclass
class ShMeasure(MeasureAsInputBase):
    measure_type: MeasureTypeEnum
    combine_type: CombinableTypeEnum
    cost: float
    year: int
    lcc: float
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    beta_target: float
    transition_level: float
    dcrest: float
