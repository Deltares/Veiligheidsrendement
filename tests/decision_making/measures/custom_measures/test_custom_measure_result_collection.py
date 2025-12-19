from vrtool.decision_making.measures.custom_measures.custom_measure_result import (
    CustomMeasureResult,
)
from vrtool.decision_making.measures.custom_measures.custom_measure_result_collection import (
    CustomMeasureResultCollection,
)
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)


class TestCustomMeasureResultCollection:
    def test_initialize(self):
        # 1. Define test data.
        _result_collection = CustomMeasureResultCollection()

        # 2. Verify expectations.
        assert isinstance(_result_collection, CustomMeasureResultCollection)
        assert isinstance(_result_collection, MeasureResultCollectionProtocol)
        assert any(_result_collection.result_collection) is False
