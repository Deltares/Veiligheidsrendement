from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_data import (
    RevetmentMeasureData,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_data_evaluator import (
    RevetmentMeasureDataBuilder,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
import pytest


class TestRevetmentMeasureDataBuilder:
    @pytest.mark.parametrize(
        "revetment_data", [pytest.param(None), pytest.param(RevetmentDataClass())]
    )
    def test_build_revetment_measure_data_collection_no_slope_parts(
        self, revetment_data: RevetmentDataClass
    ):
        # 1. Define test data.
        _crest_height = 4.2
        _target_beta = 0.4
        _transition_level = 2.4
        _evaluation_year = 2023
        _builder = RevetmentMeasureDataBuilder()

        # 2. Run test.
        _data_collection = _builder.build_revetment_measure_data_collection(
            _crest_height,
            revetment_data,
            _target_beta,
            _transition_level,
            _evaluation_year,
        )

        # 3. Verify expectations.
        assert isinstance(_data_collection, list)
        assert not any(_data_collection)

    def test_build_revetment_measure_data_collection(self):
        # 1. Define test data.
        _revetment_data = RevetmentDataClass()
        _crest_height = 4.2
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
