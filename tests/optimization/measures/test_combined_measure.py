from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import pytest

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.combined_measures.combined_measure_factory import (
    CombinedMeasureFactory,
)
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


class TestCombinedMeasure:
    @pytest.fixture(name="mocked_measure")
    def _get_valid_measure(
        self,
    ) -> Iterator[Callable[[MeasureTypeEnum, int], MeasureAsInputProtocol]]:
        @dataclass
        class MockMeasure(MeasureAsInputProtocol):
            measure_type: MeasureTypeEnum = None
            measure_result_id: int = 0
            year: int = 0
            mechanism_year_collection: MechanismPerYearProbabilityCollection = None
            l_stab_screen: float = 0.0

        def create_mocked_measure(
            measure_type: MeasureTypeEnum, measure_result_id: int
        ):
            return MockMeasure(
                measure_type=measure_type,
                measure_result_id=measure_result_id,
                mechanism_year_collection=self._get_valid_probability_collection(
                    MechanismEnum.OVERFLOW
                ),
            )

        yield create_mocked_measure

    def _get_valid_probability_collection(
        self, mechanism: MechanismEnum
    ) -> MechanismPerYearProbabilityCollection:
        _mech_per_year = MechanismPerYear(mechanism=mechanism, year=0, probability=0.5)
        return MechanismPerYearProbabilityCollection(probabilities=[_mech_per_year])

    def test_from_input(
        self, mocked_measure: Callable[[MeasureTypeEnum, int], MeasureAsInputProtocol]
    ):
        # 1. Define test data
        _primary_measure_result_id = 2
        _secondary_measure_result_id = 3
        _primary = mocked_measure(
            MeasureTypeEnum.SOIL_REINFORCEMENT, _primary_measure_result_id
        )
        _secondary = mocked_measure(
            MeasureTypeEnum.SOIL_REINFORCEMENT, _secondary_measure_result_id
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

    @pytest.mark.parametrize(
        "measure_type, expected",
        [pytest.param(MeasureTypeEnum.CUSTOM, False)]
        + [
            pytest.param(_measure_type, True)
            for _measure_type in MeasureTypeEnum
            if _measure_type not in (MeasureTypeEnum.INVALID, MeasureTypeEnum.CUSTOM)
        ],
    )
    def test_compares_to(
        self,
        measure_type: MeasureTypeEnum,
        expected: bool,
        mocked_measure: Callable[[MeasureTypeEnum, int], MeasureAsInputProtocol],
    ):
        # 1. Define test data
        _this_primary_measure_result_id = 1
        _other_primary_measure_result_id = 2
        _this_primary = mocked_measure(measure_type, _this_primary_measure_result_id)
        _other_primary = mocked_measure(measure_type, _other_primary_measure_result_id)
        _secondary = mocked_measure(measure_type, 3)

        _this_combination = CombinedMeasureFactory.from_input(
            _this_primary,
            _secondary,
            self._get_valid_probability_collection(MechanismEnum.OVERFLOW),
            7,
        )
        _other_combination = CombinedMeasureFactory.from_input(
            _other_primary,
            _secondary,
            self._get_valid_probability_collection(MechanismEnum.OVERFLOW),
            8,
        )

        # 2. Run test
        _result = _this_combination.compares_to(_other_combination)

        # 3. Verify expectations
        assert _result == expected

    @pytest.fixture(name="combined_measure_example")
    def _get_combined_measure_example_fixture(
        self, combined_measure_factory: Callable[[dict, dict], CombinedMeasureBase]
    ) -> Iterable[CombinedMeasureBase]:
        yield combined_measure_factory(
            dict(cost=4.2, base_cost=2.2, year=20),
            dict(cost=6.7, base_cost=4.2, year=0),
        )

    def test_lcc_for_sh(self, combined_measure_example: CombinedMeasureBase):
        assert combined_measure_example.lcc == pytest.approx(6.7105, 0.0001)
        assert combined_measure_example.lcc == pytest.approx(6.7220, 0.0001)
