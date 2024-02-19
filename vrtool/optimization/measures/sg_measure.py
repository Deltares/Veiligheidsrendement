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
    discount_rate: float
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    dberm: float
    dcrest: float
    start_cost: float = 0

    @property
    def lcc(self) -> float:
        return (self.cost - self.start_cost) / (1 + self.discount_rate) ** self.year

    def set_start_cost(
        self,
        previous_measure: MeasureAsInputProtocol | None,
    ):
        if self.measure_type not in [
            MeasureTypeEnum.SOIL_REINFORCEMENT,
            MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
        ]:
            return
        if (
            previous_measure is None
            or self.measure_type != previous_measure.measure_type
        ):
            if self.year == 0 and self.dberm == 0:
                self.start_cost = self.cost
                return
            raise (ValueError("First measure of type isn't zero-version"))
        self.start_cost = previous_measure.start_cost

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
