import pytest
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
    MeasureResultProtocol,
)

from vrtool.orm.io.exporters.measures.measure_result_type_converter import (
    to_measure_result_collection,
)


def test_to_measure_result_collection_given_unknown_type_raises_error():
    _measure_type = "str"
    with pytest.raises(ValueError) as exc_err:
        to_measure_result_collection(_measure_type)

    assert str(exc_err.value) == f"Unknown measure type: 'str'."


def test_to_measure_result_collection_given_dict():
    # 1. Define test data.
    _measure_as_dict = get_measure_result_as_dict()

    # 2. Run test.
    _measure_result_collection = to_measure_result_collection(_measure_as_dict)

    # 3. Verify expectations.
    assert isinstance(_measure_result_collection, MeasureResultCollectionProtocol)
    assert len(_measure_result_collection.result_collection) == 1
    validate_measure_result_as_dict(_measure_result_collection.result_collection[0])


def test_to_measure_result_collection_given_list_of_dicts():
    # 1. Define test data.
    _measure_as_list_dict = [get_measure_result_as_dict()]

    # 2. Run test.
    _measure_result_collection = to_measure_result_collection(_measure_as_list_dict)

    # 3. Verify expectations.
    assert isinstance(_measure_result_collection, MeasureResultCollectionProtocol)
    assert len(_measure_result_collection.result_collection) == 1
    validate_measure_result_as_dict(_measure_result_collection.result_collection[0])


def get_measure_result_as_dict():
    return {"ID": "just-an-id", "Cost": 4.2, "Reliability": "sth"}


def validate_measure_result_as_dict(measure_result: MeasureResultProtocol):
    _measure_as_dict = get_measure_result_as_dict()

    assert isinstance(measure_result, MeasureResultProtocol)
    assert measure_result.measure_id == _measure_as_dict["ID"]
    assert measure_result.cost == _measure_as_dict["Cost"]
    assert measure_result.section_reliability == _measure_as_dict["Reliability"]


def test_to_measure_result_given_measure_result_collection():
    # 1. Define test data.
    class MockedMeasureResultCollection(MeasureResultCollectionProtocol):
        pass

    _mocked_measure_result_collection = MockedMeasureResultCollection()

    # 2. Run test.
    _result = to_measure_result_collection(_mocked_measure_result_collection)

    # 3. Verify expectations.
    assert isinstance(_result, MeasureResultCollectionProtocol)
    assert _result == _mocked_measure_result_collection
