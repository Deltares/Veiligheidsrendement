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

    def test_get_measure_output_values_with_no_results(self):
        # 1. Define test data.
        _result_collection = CustomMeasureResultCollection()
        assert any(_result_collection.result_collection) is False

        # 2. Run test
        _output_values = _result_collection.get_measure_output_values(False, [])

        # 3. Verify expectations.
        assert _output_values == ([], [])

    def test_get_measure_output_values_with_results(self):
        # 1. Define test data.
        _result_collection = CustomMeasureResultCollection()
        _mocked_output_value = [4.2]

        class MockedMeasureResult(CustomMeasureResult):
            def get_measure_output_values(
                self, split_params: bool, beta_columns: list[str]
            ) -> tuple[list, list]:
                return _mocked_output_value, _mocked_output_value

        _result_collection.result_collection = [
            MockedMeasureResult(),
            MockedMeasureResult(),
        ]

        # 2. Run test
        (
            _input_values,
            _reliability_values,
        ) = _result_collection.get_measure_output_values(False, [])

        # 3. Verify expectations.
        _expected_values = tuple(
            ([_mocked_output_value]) * len(_result_collection.result_collection)
        )
        assert _input_values == _expected_values
        assert _reliability_values == _expected_values
