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
        _mechanisms = [_me.name for _me in MechanismEnum]
        _columns = [0, 5, 42]
        _data_values = [_idx for _idx in range(0, len(_columns))]
        _data = {_mech_name: _data_values for _mech_name in _mechanisms}
        _section_reliability.SectionReliability = DataFrame.from_dict(
            _data, orient="index", columns=list(map(str, _columns))
        )

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

    def test_get_beta_values_for_mechanisms_given_unknown_name(
        self, valid_custom_measure_result: CustomMeasureResult
    ):
        # 1. Define test data.
        _unknown_mechanism = "UnknownMechanism"
        _expected_length = len(
            valid_custom_measure_result.section_reliability.SectionReliability.columns
        )

        # 2. Run test.
        _values = valid_custom_measure_result._get_beta_values_for_mechanism(
            _unknown_mechanism
        )

        # 3. Verify expectations.
        assert len(_values) == _expected_length
        assert _values == [10.0] * _expected_length

    @pytest.mark.parametrize("split", [(True), (False)])
    def test_get_measure_output_values_with_empty_beta_columns(
        self, split: bool, valid_custom_measure_result: CustomMeasureResult
    ):
        # 1. Define test data.
        _beta_values = []

        # 2. Run test.
        (
            _input_measure,
            _output_betas,
        ) = valid_custom_measure_result.get_measure_output_values(split, _beta_values)

        # 3. Verify expectations
        assert any(_input_measure)
        assert any(_output_betas) is False

    def test_get_measure_output_values(
        self, valid_custom_measure_result: CustomMeasureResult
    ):
        # 1. Define test data.
        _beta_values = [MechanismEnum.OVERFLOW.name, MechanismEnum.PIPING.name]

        # 2. Run test.
        (
            _input_values,
            _output_betas,
        ) = valid_custom_measure_result.get_measure_output_values(False, _beta_values)

        # 3. Verify expectations.
        assert any(_input_values)
        assert any(_output_betas)
        assert _output_betas == [0, 1, 2, 0, 1, 2]
