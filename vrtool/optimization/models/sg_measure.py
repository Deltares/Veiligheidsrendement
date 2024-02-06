from __future__ import annotations

from dataclasses import dataclass

from measure_as_input_base import MeasureAsInputBase, MechanismPerYearProbabilityCollection

@dataclass
class SgMeasure(MeasureAsInputBase):
    measure_type: int # TODO enum
    combine_type: int # TODO enum
    cost: float
    year: int
    lcc: float
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    dberm: float
    dcrest: float
