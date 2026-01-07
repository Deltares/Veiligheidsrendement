from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultProtocol,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_section_reliability import (
    RevetmentMeasureSectionReliability,
)


class TestRevetmentMeasureSectionReliability:
    def test_init(self):
        _measure_reliability = RevetmentMeasureSectionReliability()
        assert isinstance(_measure_reliability, RevetmentMeasureSectionReliability)
        assert isinstance(_measure_reliability, MeasureResultProtocol)
