from dataclasses import dataclass

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


@dataclass(kw_only=True)
class MeasureAsInputBase(MeasureAsInputProtocol):
    """
    (Base) class introduced to reduce code duplication at the
    `ShMeasure`, `SgMeasure` and `ShSgMeasure`.
    """

    measure_result_id: int
    measure_type: MeasureTypeEnum
    combine_type: CombinableTypeEnum
    cost: float
    base_cost: float
    discount_rate: float
    year: int
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    l_stab_screen: float
