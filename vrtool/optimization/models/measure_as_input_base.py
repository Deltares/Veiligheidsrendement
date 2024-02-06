from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mechanism_per_year_probability_collection import MechanismPerYearProbabilityCollection

@runtime_checkable
@dataclass
class MeasureAsInputBase(Protocol):
    """stores data for measure in optimization"""

    measure_type: int # TODO enum
    combine_type: int # TODO enum
    cost: float
    year: int
    lcc: float
    mechanism_year_collection: MechanismPerYearProbabilityCollection
