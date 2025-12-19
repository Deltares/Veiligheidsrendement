import pytest
from pandas import DataFrame

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.custom_measures.custom_measure_result import (
    CustomMeasureResult,
)
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultProtocol,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class TestCustomMeasureResult:
    def test_initialize(self):
        # 1. Define test data.
        _measure_result = CustomMeasureResult()

        # 2. Verify expectations.
        assert isinstance(_measure_result, CustomMeasureResult)
        assert isinstance(_measure_result, MeasureResultProtocol)

    @pytest.fixture(name="valid_custom_measure_result")
    def _get_valid_custom_measure_result_fixture(self) -> CustomMeasureResult:
        _measure_result = CustomMeasureResult()
        _measure_result.beta_target = 4.2
        _measure_result.cost = 240000
        _measure_result.measure_id = "1"
        _measure_result.measure_name = "TestMeasureResult"
        _measure_result.measure_year = 0
        _measure_result.reinforcement_type = MeasureTypeEnum.CUSTOM.name
        _measure_result.combinable_type = CombinableTypeEnum.FULL.name

        # Define section reliability
        _section_reliability = SectionReliability()
        _mechanisms = [_me for _me in MechanismEnum]
        _data_values = {0: 0.1, 5: 0.05, 42: 0.01}
        for _mech in _mechanisms:
            _section_reliability.set_reliabilities_for_mechanism(_mech, _data_values)

        _measure_result.section_reliability = _section_reliability

        return _measure_result

    def test_get_input_vector_without_splitting(
        self, valid_custom_measure_result: CustomMeasureResult
    ):
        # 1. Run test.
        _input_vector = valid_custom_measure_result._get_input_vector(False)

        # 2. Verify expectations.
        assert _input_vector == [
            valid_custom_measure_result.measure_id,
            valid_custom_measure_result.reinforcement_type,
            valid_custom_measure_result.combinable_type,
            valid_custom_measure_result.measure_year,
            -999,
            valid_custom_measure_result.cost,
        ]

    def test_get_input_vector_with_splitting(
        self, valid_custom_measure_result: CustomMeasureResult
    ):
        # 1. Run test.
        _input_vector = valid_custom_measure_result._get_input_vector(True)

        # 2. Verify expectations.
        assert _input_vector == [
            valid_custom_measure_result.measure_id,
            valid_custom_measure_result.reinforcement_type,
            valid_custom_measure_result.combinable_type,
            valid_custom_measure_result.measure_year,
            -999,
            -999,
            -999,
            -999,
            -999,
            -999,
            valid_custom_measure_result.cost,
        ]
