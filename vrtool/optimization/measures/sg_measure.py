import math
from dataclasses import dataclass

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.measure_as_input_base import MeasureAsInputBase


@dataclass(kw_only=True)
class SgMeasure(MeasureAsInputBase):
    """
    Class to represent measures that do not have a crest component.
    """

    dberm: float

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
    def is_combinable_type_allowed(combinable_type: CombinableTypeEnum) -> bool:
        return combinable_type in [
            CombinableTypeEnum.FULL,
            CombinableTypeEnum.COMBINABLE,
            CombinableTypeEnum.PARTIAL,
        ]

    def is_initial_measure(self) -> bool:
        if self.year != 0:
            return False
        return math.isclose(self.dberm, 0) or math.isnan(self.dberm)
