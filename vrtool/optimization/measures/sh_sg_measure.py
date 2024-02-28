from dataclasses import dataclass

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


@dataclass
class ShSgMeasure(MeasureAsInputProtocol):
    """
    Class to represent soil measures that have both a crest and berm component.
    These are used to store the opimization result for the aggregated Sh/Sg combined measures.
    """

    measure_type: MeasureTypeEnum
    combine_type: CombinableTypeEnum
    measure_result_id: int
    dcrest: float
    dberm: float
    cost: float = 0
    discount_rate: float = 0
    year: int = 0
    mechanism_year_collection: MechanismPerYearProbabilityCollection = (
        MechanismPerYearProbabilityCollection([])
    )
    start_cost: float = 0
    lcc: float = 0

    @staticmethod
    def get_concrete_parameters() -> list[str]:
        return ["dberm", "dcrest"]

    @staticmethod
    def is_mechanism_allowed(mechanism: MechanismEnum) -> bool:
        return False

    @staticmethod
    def get_allowed_mechanisms() -> list[MechanismEnum]:
        return []

    @staticmethod
    def get_allowed_measure_combinations() -> (
        dict[CombinableTypeEnum, list[CombinableTypeEnum | None]]
    ):
        return {}

    def is_initial_cost_measure(self) -> bool:
        return False
