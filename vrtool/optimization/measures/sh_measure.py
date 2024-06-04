import logging
import math
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
    """
    Class to represent measures that do not have a berm component.
    """

    measure_type: MeasureTypeEnum
    combine_type: CombinableTypeEnum
    measure_result_id: int
    cost: float
    discount_rate: float
    year: int
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    beta_target: float
    transition_level: float
    dcrest: float
    l_stab_screen: float

    @property
    def lcc(self) -> float:
        """
        Value for the `life-cycle-cost` of this measure.
        When the `dcrest` is the "initial" value (`0` or `-999`) and there is no stability screen,
        the cost will be computed as `0`.

        Returns:
            float: The calculated lcc.
        """
        if self.measure_type != MeasureTypeEnum.CUSTOM:
            if self.dcrest in [0, -999] and math.isnan(self.l_stab_screen):
                return 0

        return (self.cost - self.start_cost) / (1 + self.discount_rate) ** self.year

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

    @staticmethod
    def is_combinable_type_allowed(combinable_type: CombinableTypeEnum) -> bool:
        return combinable_type in [
            CombinableTypeEnum.FULL,
            CombinableTypeEnum.COMBINABLE,
            CombinableTypeEnum.REVETMENT,
        ]

    def is_initial_cost_measure(self) -> bool:
        if self.year != 0:
            return False

        return math.isnan(self.l_stab_screen) and (
            math.isclose(self.dcrest, 0) or math.isnan(self.dcrest)
        )
