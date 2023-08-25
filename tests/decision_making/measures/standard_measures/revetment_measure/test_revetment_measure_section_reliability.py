import pandas as pd

from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_section_reliability import (
    RevetmentMeasureSectionReliability,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class TestRevetmentMeasureSectionReliability:
    def test_init(self):
        _measure_reliability = RevetmentMeasureSectionReliability()
        assert isinstance(_measure_reliability, RevetmentMeasureSectionReliability)
        assert isinstance(_measure_reliability, MeasureResultCollectionProtocol)

    def test_get_measure_output_values_with_split_param(self):
        # 1. Define test data.
        _measure_reliability = RevetmentMeasureSectionReliability()
        _measure_reliability.measure_id = "Lorem ipsum"
        _measure_reliability.measure_name = "Nostrud qui laboris"
        _measure_reliability.reinforcement_type = "Mollit exercitation"
        _measure_reliability.combinable_type = "amet magna"
        _measure_reliability.beta_target = 4.2
        _measure_reliability.transition_level = 2.4
        _measure_reliability.cost = 42
        _measure_reliability.measure_year = 1984
        _measure_reliability.section_reliability = SectionReliability()
        _data_dict = {"ABC": [1, 2, 3], "DEF": [4, 5, 6], "GHI": [7, 8, 9]}
        _measure_reliability.section_reliability.SectionReliability = (
            pd.DataFrame.from_dict(_data_dict, orient="index")
        )

        _expected_measure_input = [
            _measure_reliability.measure_id,
            _measure_reliability.reinforcement_type,
            _measure_reliability.combinable_type,
            _measure_reliability.measure_year,
            "yes",
            -999,
            -999,
            _measure_reliability.beta_target,
            _measure_reliability.transition_level,
            _measure_reliability.cost,
        ]

        _expected_betas_vector = [7, 8, 9, 1, 2, 3, 4, 5, 6]

        # 2. Run test.
        _results = _measure_reliability.get_measure_output_values(
            True, ["GHI", "ABC", "DEF"]
        )

        # 3. Verify expectations.
        assert isinstance(_results, tuple)
        assert all(isinstance(_result, list) for _result in _results)
        assert _results[0] == _expected_measure_input
        assert _results[1] == _expected_betas_vector
