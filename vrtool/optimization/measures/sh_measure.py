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
    lcc: float
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    beta_target: float
    transition_level: float
    dcrest: float

    @classmethod
    def is_mechanism_allowed(cls, mechanism: MechanismEnum) -> bool:
        return mechanism in cls.get_allowed_mechanisms()

    @classmethod
    def get_allowed_mechanisms(cls) -> list[MechanismEnum]:
        return [MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT]

    @classmethod
    def get_allowed_combine_measure_types(cls) -> list[MeasureTypeEnum]:
        return [MeasureTypeEnum.REVETMENT]

    def __post_init__(self):
        """
        Set LCC to 0 for Sh to avoid double counting with Sg
        """
        if self.measure_type in [
            MeasureTypeEnum.DIAPHRAGM_WALL,
            MeasureTypeEnum.STABILITY_SCREEN,
            MeasureTypeEnum.VERTICAL_GEOTEXTILE,
        ]:
            self.lcc = 0
