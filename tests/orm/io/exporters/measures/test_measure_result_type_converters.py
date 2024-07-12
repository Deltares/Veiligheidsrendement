import pytest

from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
    MeasureResultProtocol,
)
from vrtool.orm.io.exporters.measures.measure_result_type_converter import (
    filter_supported_parameters_dict,
    to_measure_result_collection,
)

unsupported_dict_cases = [
    pytest.param(dict(), id="Empty dict"),
    pytest.param(dict(unsupported_prop=4.2), id="Unsupported entries dict"),
]


class TestFilterSupportedParametersDict:
    @pytest.mark.parametrize(
        "input_dict",
        unsupported_dict_cases,
    )
    def test_given_dict_with_unsupported_params_then_returns_empty_dict(
        self,
        input_dict: dict,
    ):
        assert filter_supported_parameters_dict(input_dict) == dict()

    @pytest.mark.parametrize("unsupported_dict", unsupported_dict_cases)
    @pytest.mark.parametrize(
        "supported_dict",
        [
            pytest.param(dict(dberm=4.2), id="With dberm"),
            pytest.param(dict(dcrest=2.4), id="With dcrest"),
            pytest.param(dict(dberm=4.2, dcrest=2.4), id="With dberm and dcrest"),
        ],
    )
    def test_given_dict_with_supported_params_returns_only_supported_params(
        self, supported_dict: dict, unsupported_dict: dict
    ):
        # 1. Define test data
        _input_dict = supported_dict | unsupported_dict

        # 2. Run test.
        _result_dict = filter_supported_parameters_dict(_input_dict)

        # 3. Verify expectations.
        assert _result_dict == supported_dict


class TestToMeasureResultCollection:
    def test_to_measure_result_collection_given_unknown_type_raises_error(self):
        _measure_type = "str"
        with pytest.raises(ValueError) as exc_err:
            to_measure_result_collection(_measure_type)

        assert str(exc_err.value) == "Unknown measure type: 'str'."

    def test_to_measure_result_collection_given_dict(self):
        # 1. Define test data.
        _measure_as_dict = self.get_measure_result_as_dict()

        # 2. Run test.
        _measure_result_collection = to_measure_result_collection(_measure_as_dict)

        # 3. Verify expectations.
        assert isinstance(_measure_result_collection, MeasureResultCollectionProtocol)
        assert len(_measure_result_collection.result_collection) == 1
        self.validate_measure_result_as_dict(
            _measure_result_collection.result_collection[0]
        )

    def test_to_measure_result_collection_given_list_of_dicts(self):
        # 1. Define test data.
        _measure_as_list_dict = [self.get_measure_result_as_dict()]

        # 2. Run test.
        _measure_result_collection = to_measure_result_collection(_measure_as_list_dict)

        # 3. Verify expectations.
        assert isinstance(_measure_result_collection, MeasureResultCollectionProtocol)
        assert len(_measure_result_collection.result_collection) == 1
        self.validate_measure_result_as_dict(
            _measure_result_collection.result_collection[0]
        )

    def get_measure_result_as_dict(self):
        return {"ID": "just-an-id", "Cost": 4.2, "Reliability": "sth"}

    def validate_measure_result_as_dict(self, measure_result: MeasureResultProtocol):
        _measure_as_dict = self.get_measure_result_as_dict()

        assert isinstance(measure_result, MeasureResultProtocol)
        assert measure_result.measure_id == _measure_as_dict["ID"]
        assert measure_result.cost == _measure_as_dict["Cost"]
        assert measure_result.section_reliability == _measure_as_dict["Reliability"]

    def test_to_measure_result_given_measure_result_collection(self):
        # 1. Define test data.
        class MockedMeasureResultCollection(MeasureResultCollectionProtocol):
            pass

        _mocked_measure_result_collection = MockedMeasureResultCollection()

        # 2. Run test.
        _result = to_measure_result_collection(_mocked_measure_result_collection)

        # 3. Verify expectations.
        assert isinstance(_result, MeasureResultCollectionProtocol)
        assert _result == _mocked_measure_result_collection
