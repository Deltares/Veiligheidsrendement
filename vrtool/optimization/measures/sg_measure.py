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
class SgMeasure(MeasureAsInputProtocol):
    measure_type: MeasureTypeEnum
    combine_type: CombinableTypeEnum
    measure_result_id: int
    cost: float
    year: int
    discount_rate: float
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    dberm: float
    dcrest: float
    _start_cost: float = 0

    @property
    def lcc(self) -> float:
        return (self.cost - self.start_cost) / (1 + self.discount_rate) ** self.year

    @property
    def start_cost(self) -> float:
        """
        Gets the initial cost for this measure. This is a "protected" property as its
        value depends on which other measures are present as well as its measure type
        (`MeasureTypeEnum`).

        Returns:
            float: The start cost value.
        """
        return self._start_cost

    @start_cost.setter
    def start_cost(self, value: float):
        if self.measure_type not in [
            MeasureTypeEnum.SOIL_REINFORCEMENT,
            MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
        ]:
            logging.info(
                "Start cost for {} must be always 0. (Attempt to set to {}).".format(
                    self.measure_type, value
                )
            )
            value = 0
        self._start_cost = value

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
    def get_concrete_parameters() -> list[str]:
        return ["dberm", "dcrest"]

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

    def is_initial_cost_measure(self) -> bool:
        if self.year != 0:
            return False

        return math.isclose(self.dberm, 0) or math.isnan(self.dberm)
