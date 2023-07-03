from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.revetment_measure import (
    RevetmentMeasure,
)


class TestRevetmentMeasure:
    def test_initialize(self):
        # Test to validate it's a parameterless constructor.
        _revetment_measure = RevetmentMeasure()
        assert isinstance(_revetment_measure, RevetmentMeasure)
        assert isinstance(_revetment_measure, MeasureProtocol)
