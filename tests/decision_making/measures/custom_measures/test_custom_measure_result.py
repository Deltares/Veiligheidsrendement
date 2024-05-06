from vrtool.decision_making.measures.custom_measures.custom_measure_result import (
    CustomMeasureResult,
)
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultProtocol,
)


class TestCustomMeasureResult:
    def test_initialize(self):
        # 1. Define test data.
        _measure_result = CustomMeasureResult()

        # 3. Verify expectations.
        assert isinstance(_measure_result, CustomMeasureResult)
        assert isinstance(_measure_result, MeasureResultProtocol)
