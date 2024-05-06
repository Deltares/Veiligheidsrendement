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

    def test_get_measure_output_values_with_no_results(self):
        # 1. Define test data.
        _result_collection = CustomMeasureResultCollection()
        assert any(_result_collection.result_collection) is False

        # 2. Run test
        _output_values = _result_collection.get_measure_output_values(False, [])

        # 3. Verify expectations.
        assert _output_values == ([], [])
