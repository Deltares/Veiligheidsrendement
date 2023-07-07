from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_data import (
    RevetmentMeasureData,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_data_evaluator import (
    RevetmentMeasureDataBuilder,
)


class TestRevetmentMeasureDataBuilder:
    def test_build_revetment_measure_data_collection(self):
        # 1. Define test data.
        _crest_height = 4.2
        _revetment_data = None
        _target_beta = 0.4
        _transition_level = 2.4
        _evaluation_year = 2023
        _builder = RevetmentMeasureDataBuilder()

        # 2. Run test.
        _data_collection = _builder.build_revetment_measure_data_collection(
            _crest_height,
            _revetment_data,
            _target_beta,
            _transition_level,
            _evaluation_year,
        )

        # 3. Verify expectations.
        assert isinstance(_data_collection, list)
        assert all(
            isinstance(RevetmentMeasureData, _data) for _data in _data_collection
        )
