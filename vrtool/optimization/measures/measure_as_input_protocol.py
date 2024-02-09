from typing import Protocol, runtime_checkable

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


@runtime_checkable
class MeasureAsInputProtocol(Protocol):
    """stores data for measure in optimization"""

    measure_type: MeasureTypeEnum
    combine_type: CombinableTypeEnum
    cost: float
    year: int
    lcc: float
    mechanism_year_collection: MechanismPerYearProbabilityCollection

    @classmethod
    def is_mechanism_allowed(cls, mechanism: MechanismEnum) -> bool:
        pass

    @classmethod
    def get_allowed_mechanisms(cls) -> list[MechanismEnum]:
        pass

    @classmethod
    def get_allowed_combinable_types(cls) -> list[CombinableTypeEnum]:
        pass
