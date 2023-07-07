import pytest
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result_collection import (
    RevetmentMeasureResultCollection,
)
from itertools import product


class TestRevetmentMeasureResultCollection:
    def test_init(self):
        _result_collection = RevetmentMeasureResultCollection()
        assert isinstance(_result_collection, RevetmentMeasureResultCollection)
        assert isinstance(_result_collection, MeasureResultCollectionProtocol)

    def test_get_measure_output_values_with_split_param(self):
        # 1. Define test data.
        _result_collection = RevetmentMeasureResultCollection()
        _result_collection.measure_id = "Lorem ipsum"
        _result_collection.reinforcement_type = "Mollit exercitation"
        _result_collection.combinable_type = "amet magna"
        _beta_targets = [0.24, 0.42, 0.89]
        _transition_levels = [2.4, 4.2]
        _years = [1984, 2001, 2049]
        _possibilities_matrix = list(
            product(*[_beta_targets, _transition_levels, _years])
        )
        _result_collection.revetment_measure_results = [
            RevetmentMeasureResult(
                year=possibility[2],
                beta_target=possibility[0],
                beta_combined=10 * _years.index(possibility[2]),
                transition_level=possibility[1],
                cost=24,
                revetment_measures=[],
            )
            for possibility in _possibilities_matrix
        ]

        _expected_measure_input = [
            _result_collection.measure_id,
            _result_collection.reinforcement_type,
            _result_collection.combinable_type,
            1984,
            "yes",
            -999,
            -999,
        ]
        _expected_measure_input_vector = [
            _expected_measure_input + [0.24, 2.4, 24],
            _expected_measure_input + [0.24, 4.2, 24],
            _expected_measure_input + [0.42, 2.4, 24],
            _expected_measure_input + [0.42, 4.2, 24],
            _expected_measure_input + [0.89, 2.4, 24],
            _expected_measure_input + [0.89, 4.2, 24],
        ]
        _expected_betas_vector = [[0, 10, 20]] * (
            len(_beta_targets) * len(_transition_levels)
        )
        # 2. Run test.
        _results = _result_collection.get_measure_output_values(True)

        # 3. Verify expectations.
        assert isinstance(_results, tuple)
        assert all(isinstance(_result, list) for _result in _results)
        assert len(_results[0]) == len(_results[1])
        assert _results[0] == _expected_measure_input_vector
        assert _results[1] == _expected_betas_vector
