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
class ShMeasure(MeasureAsInputProtocol):
    measure_type: MeasureTypeEnum
    combine_type: CombinableTypeEnum
    cost: float
    discount_rate: float
    year: int
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    beta_target: float
    transition_level: float
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
            MeasureTypeEnum.VERTICAL_GEOTEXTILE,
            MeasureTypeEnum.DIAPHRAGM_WALL,
            MeasureTypeEnum.STABILITY_SCREEN,
        ]:
            return
        if (
            previous_measure is None
            or self.measure_type != previous_measure.measure_type
        ):
            if self.year == 0 and self.dcrest in [0, -999]:
                self.start_cost = self.cost
                return
            raise (ValueError("First measure of type isn't zero-version"))
        self.start_cost = previous_measure.start_cost

    @staticmethod
    def is_mechanism_allowed(mechanism: MechanismEnum) -> bool:
        return mechanism in ShMeasure.get_allowed_mechanisms()

    @staticmethod
    def get_allowed_mechanisms() -> list[MechanismEnum]:
        return [MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT]

    @staticmethod
    def get_allowed_measure_combinations() -> (
        dict[CombinableTypeEnum, list[CombinableTypeEnum | None]]
    ):
        return {
            CombinableTypeEnum.COMBINABLE: [None, CombinableTypeEnum.REVETMENT],
            CombinableTypeEnum.FULL: [None, CombinableTypeEnum.REVETMENT],
        }
