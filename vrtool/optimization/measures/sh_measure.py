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
    year: int
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    beta_target: float
    transition_level: float
    dcrest: float
    _lcc: float = 0

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
            CombinableTypeEnum.REVETMENT: [None],
            CombinableTypeEnum.COMBINABLE: [None, CombinableTypeEnum.REVETMENT],
            CombinableTypeEnum.FULL: [None, CombinableTypeEnum.REVETMENT],
        }

    @property
    def lcc(self) -> float:
        if self.measure_type in [
            MeasureTypeEnum.DIAPHRAGM_WALL,
            MeasureTypeEnum.STABILITY_SCREEN,
            MeasureTypeEnum.VERTICAL_GEOTEXTILE,
        ]:
            return 0
        return self._lcc

    @lcc.setter
    def lcc(self, value: float):
        self._lcc = value
