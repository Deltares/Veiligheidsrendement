from dataclasses import dataclass

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.measure_as_input_base import MeasureAsInputBase
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


@dataclass(kw_only=True)
class ShSgMeasure(MeasureAsInputBase):
    """
    Class to represent soil measures that have both a crest and berm component.
    These are used to store the optimization result for the aggregated Sh/Sg combined measures.
    """

    cost: float = 0
    base_cost: float = 0
    discount_rate: float = 0
    year: int = 0
    mechanism_year_collection: MechanismPerYearProbabilityCollection = (
        MechanismPerYearProbabilityCollection([])
    )
    l_stab_screen: float
    dcrest: float
    dberm: float

    def is_base_measure(self) -> bool:
        return False

    @staticmethod
    def is_mechanism_allowed(mechanism: MechanismEnum) -> bool:
        return mechanism in ShSgMeasure.get_allowed_mechanisms()

    @staticmethod
    def get_allowed_mechanisms() -> list[MechanismEnum]:
        return []

    @staticmethod
    def get_allowed_measure_combinations() -> (
        dict[CombinableTypeEnum, list[CombinableTypeEnum | None]]
    ):
        return {}
