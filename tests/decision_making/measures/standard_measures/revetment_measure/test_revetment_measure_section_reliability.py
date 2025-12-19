import pandas as pd

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultProtocol,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_section_reliability import (
    RevetmentMeasureSectionReliability,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class TestRevetmentMeasureSectionReliability:
    def test_init(self):
        _measure_reliability = RevetmentMeasureSectionReliability()
        assert isinstance(_measure_reliability, RevetmentMeasureSectionReliability)
        assert isinstance(_measure_reliability, MeasureResultProtocol)
