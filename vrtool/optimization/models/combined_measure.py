from __future__ import annotations

from dataclasses import dataclass

from measure_as_input_base import MeasureAsInputBase, MechanismPerYearProbabilityCollection

@dataclass
class CombinedMeasure:
    primary: MeasureAsInputBase
    secondary: MeasureAsInputBase
    mechanism_year_collection: MechanismPerYearProbabilityCollection
