from dataclasses import dataclass

import pytest

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


class TestCombinedMeasure:
    @dataclass
    class MockMeasure(MeasureAsInputProtocol):
        measure_type: MeasureTypeEnum = None
        measure_result_id: int = 0
        year: int = 0
        mechanism_year_collection: MechanismPerYearProbabilityCollection = None
        l_stab_screen: float = 0.0

    def _get_valid_measure(
        self, measure_type: MeasureTypeEnum, measure_result_id: int
    ) -> MeasureAsInputProtocol:
        return self.MockMeasure(
            measure_type=measure_type,
            measure_result_id=measure_result_id,
            mechanism_year_collection=self._get_valid_probability_collection(
                MechanismEnum.OVERFLOW
            ),
        )

    def _get_valid_probability_collection(
        self, mechanism: MechanismEnum
    ) -> MechanismPerYearProbabilityCollection:
        _mech_per_year = MechanismPerYear(mechanism=mechanism, year=0, probability=0.5)
        return MechanismPerYearProbabilityCollection(probabilities=[_mech_per_year])

    def test_from_input(self):
        # 1. Define test data
        _primary_measure_result_id = 2
        _secondary_measure_result_id = 3
        _primary = self._get_valid_measure(
            MeasureTypeEnum.SOIL_REINFORCEMENT, _primary_measure_result_id
        )
        _secondary = self._get_valid_measure(
            MeasureTypeEnum.SOIL_REINFORCEMENT, _secondary_measure_result_id
        )
        _sequence_nr = 7

        # 2. Run test
        _combination = CombinedMeasure.from_input(
            _primary,
            _secondary,
            self._get_valid_probability_collection(MechanismEnum.OVERFLOW),
            _sequence_nr,
        )

        # 3. Verify expectations
        assert _combination.primary.measure_result_id == _primary_measure_result_id
        assert _combination.secondary.measure_result_id == _secondary_measure_result_id
        assert _combination.sequence_nr == 7

    @pytest.mark.parametrize(
        "measure_type, expected",
        [[MeasureTypeEnum.SOIL_REINFORCEMENT, True], [MeasureTypeEnum.CUSTOM, False]],
    )
    def test_compares_to(self, measure_type: MeasureTypeEnum, expected: bool):
        # 1. Define test data
        _this_primary_measure_result_id = 1
        _other_primary_measure_result_id = 2
        _this_primary = self._get_valid_measure(
            measure_type, _this_primary_measure_result_id
        )
        _other_primary = self._get_valid_measure(
            measure_type, _other_primary_measure_result_id
        )
        _secondary = self._get_valid_measure(measure_type, 3)

        _this_combination = CombinedMeasure.from_input(
            _this_primary,
            _secondary,
            self._get_valid_probability_collection(MechanismEnum.OVERFLOW),
            7,
        )
        _other_combination = CombinedMeasure.from_input(
            _other_primary,
            _secondary,
            self._get_valid_probability_collection(MechanismEnum.OVERFLOW),
            8,
        )

        # 2. Run test
        _result = _this_combination.compares_to(_other_combination)

        # 3. Verify expectations
        assert _result == expected
