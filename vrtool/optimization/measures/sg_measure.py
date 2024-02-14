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
class SgMeasure(MeasureAsInputProtocol):
    measure_type: MeasureTypeEnum
    combine_type: CombinableTypeEnum
    cost: float
    year: int
    lcc: float
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    dberm: float
    dcrest: float

    @staticmethod
    def is_mechanism_allowed(mechanism: MechanismEnum) -> bool:
        return mechanism in SgMeasure.get_allowed_mechanisms()

    @staticmethod
    def get_allowed_mechanisms() -> list[MechanismEnum]:
        return [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING]

    @staticmethod
    def get_allowed_measure_combinations() -> (
        dict[CombinableTypeEnum, list[CombinableTypeEnum | None]]
    ):
        return {
            CombinableTypeEnum.COMBINABLE: [None, CombinableTypeEnum.PARTIAL],
            CombinableTypeEnum.FULL: [None],
        }

    @staticmethod
    def get_start_cost(
        start_cost_dict: dict[MeasureTypeEnum, float],
        measure_type: MeasureTypeEnum,
        year: int,
        dberm: float,
        cost: float,
    ) -> float:
        if measure_type not in [
            MeasureTypeEnum.SOIL_REINFORCEMENT,
            MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
        ]:
            return 0
        if measure_type in start_cost_dict.keys():
            return start_cost_dict[measure_type]
        if year == 0 and dberm == 0:
            start_cost_dict[measure_type] = cost
            return cost
        return 0
