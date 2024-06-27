from typing import Callable

import pytest

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.combined_measures.combined_measure_factory import (
    CombinedMeasureFactory,
)
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)


class TestCombinedMeasureFactory:
    def test_from_input(
        self,
    ):
        pytest.fail("todo")
        # 1. Define test data
        _primary_measure_result_id = 2
        _secondary_measure_result_id = 3
        _primary = mocked_measure(
            measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
            measure_result_id=_primary_measure_result_id,
        )
        _secondary = mocked_measure(
            measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
            measure_result_id=_secondary_measure_result_id,
        )
        _sequence_nr = 7

        # 2. Run test
        _combination = CombinedMeasureFactory.from_input(
            _primary,
            _secondary,
            self._get_valid_probability_collection(MechanismEnum.OVERFLOW),
            _sequence_nr,
        )

        # 3. Verify expectations
        assert _combination.primary.measure_result_id == _primary_measure_result_id
        assert _combination.secondary.measure_result_id == _secondary_measure_result_id
        assert _combination.sequence_nr == 7
