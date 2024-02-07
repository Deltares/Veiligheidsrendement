from typing import Protocol, runtime_checkable

from vrtool.optimization.measures.mechanism_per_year_probability_collection import MechanismPerYearProbabilityCollection
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum

@runtime_checkable
class MeasureAsInputProtocol(Protocol):
    """stores data for measure in optimization"""

    measure_type: MeasureTypeEnum
    combine_type: CombinableTypeEnum
    cost: float
    year: int
    lcc: float
    mechanism_year_collection: MechanismPerYearProbabilityCollection
