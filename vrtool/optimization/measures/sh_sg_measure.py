import math
from dataclasses import dataclass

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.combined_measure import CombinedMeasure
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
    These are used to store the optimization result for the aggregated Sh/Sg combined measures.
    """

    measure_type: MeasureTypeEnum
    combine_type: CombinableTypeEnum
    measure_result_id: int
    dcrest: float
    dberm: float
    l_stab_screen: float
    cost: float = 0
    start_cost: float = 0
    discount_rate: float = 0
    year: int = 0
    mechanism_year_collection: MechanismPerYearProbabilityCollection = (
        MechanismPerYearProbabilityCollection([])
    )
    lcc: float = 0

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

    def matches_with_sh_sg_measure(
        self, sh_combination: CombinedMeasure, sg_combination: CombinedMeasure
    ) -> bool:
        """
        Compares the instance of this `ShSgMeasure` with two combined measure
        representing a `ShMeasure` and `SgMeasure` each. When their `dcrest`
        `dberm`, `l_stab_screen` and `measure_type` match this `ShSgMeasure`
        we consider them as equal.

        Args:
            sh_combination (CombinedMeasure): Combined `ShMeasure`.
            sg_combination (CombinedMeasure): Combined `SgMeasure`.

        Returns:
            bool: Comparison result.
        """

        def floats_are_equal_or_nan(left_float: float, right_float: float) -> bool:
            """
            Compares two floats for equality, when both are `float("nan")` then
            we considered them as equal.
            """
            if math.isnan(left_float) and math.isnan(right_float):
                return True
            return left_float == right_float

        return (
            floats_are_equal_or_nan(self.dcrest, sh_combination.primary.dcrest)
            and floats_are_equal_or_nan(self.dberm, sg_combination.primary.dberm)
            and floats_are_equal_or_nan(
                self.l_stab_screen, sg_combination.primary.l_stab_screen
            )
            and self.measure_type == sh_combination.primary.measure_type
        )
