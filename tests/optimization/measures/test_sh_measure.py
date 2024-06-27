from typing import Callable, Iterable

import pytest

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.measure_as_input_base import MeasureAsInputBase
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.sh_measure import ShMeasure


class TestShMeasure:
    @pytest.fixture(name="create_sh_measure")
    def _get_sh_measure_factory(
        self,
    ) -> Iterable[Callable[[MeasureTypeEnum, CombinableTypeEnum], ShMeasure]]:
        def create_sh_measure(
            measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
        ) -> ShMeasure:
            return ShMeasure(
                measure_result_id=42,
                measure_type=measure_type,
                combine_type=combinable_type,
                cost=10.5,
                base_cost=4.2,
                year=10,
                discount_rate=0.03,
                mechanism_year_collection=None,
                beta_target=1.1,
                transition_level=0.5,
                dcrest=0.1,
                l_stab_screen=float("nan"),
            )

        yield create_sh_measure

    def test_create_sh_measure(
        self,
        create_sh_measure: Callable[[MeasureTypeEnum, CombinableTypeEnum], ShMeasure],
    ):
        # 1. Define input
        _measure_type = MeasureTypeEnum.DIAPHRAGM_WALL
        _combine_type = CombinableTypeEnum.FULL

        # 2. Run test
        _measure = create_sh_measure(_measure_type, _combine_type)

        # 3. Verify expectations
        assert isinstance(_measure, ShMeasure)
        assert isinstance(_measure, MeasureAsInputBase)
        assert isinstance(_measure, MeasureAsInputProtocol)
        assert _measure.measure_type == _measure_type
        assert _measure.combine_type == _combine_type
        assert _measure.cost == pytest.approx(10.5)
        assert _measure.base_cost == pytest.approx(4.2)
        assert _measure.year == 10
        assert _measure.discount_rate == pytest.approx(0.03)
        assert _measure.mechanism_year_collection is None
        assert _measure.beta_target == pytest.approx(1.1)
        assert _measure.transition_level == pytest.approx(0.5)
        assert _measure.dcrest == pytest.approx(0.1)

    @pytest.mark.parametrize(
        "mechanism, expected",
        [
            pytest.param(MechanismEnum.OVERFLOW, True, id="VALID OVERFLOW"),
            pytest.param(MechanismEnum.PIPING, False, id="INVALID PIPING"),
        ],
    )
    def test_is_mechanism_allowed(self, mechanism: MechanismEnum, expected: bool):
        # 1./2. Define input & Run test
        _results = ShMeasure.is_mechanism_allowed(mechanism)

        # 3. Verify expectations
        assert _results == expected

    def test_get_allowed_mechanism(self):
        # 1./2. Define input & Run test
        _expected_mechanisms = [MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT]
        _allowed_mechanisms = ShMeasure.get_allowed_mechanisms()

        # 3. Verify expectations
        assert len(_expected_mechanisms) == len(_allowed_mechanisms)
        assert all(x in _expected_mechanisms for x in _allowed_mechanisms)

    def test_get_allowed_measure_combination(self):
        # 1./2. Define input & Run test
        _allowed_combinations = ShMeasure.get_allowed_measure_combinations()

        # 3. Verify expectations
        assert isinstance(_allowed_combinations, dict)
        assert _allowed_combinations

    @pytest.mark.parametrize(
        "year, dcrest, expected_result",
        [
            pytest.param(1, float("nan"), False, id="year != 0"),
            pytest.param(0, 4.2, False, id="year == 0; dcrest is > 0"),
            pytest.param(0, 0, True, id="year == 0; dcrest = 0"),
            pytest.param(0, float("nan"), True, id="year == 0; dcrest == 'nan'"),
        ],
    )
    def test_is_base_measure_returns_expectation(
        self,
        year: int,
        dcrest: float,
        expected_result: bool,
        create_sh_measure: Callable[[MeasureTypeEnum, CombinableTypeEnum], ShMeasure],
    ):
        # 1. Define test data.
        # Measure and Combinable TypeEnum should not matter for this test.
        _sh_measure = create_sh_measure(
            MeasureTypeEnum.INVALID, CombinableTypeEnum.INVALID
        )
        _sh_measure.year = year
        _sh_measure.dcrest = dcrest

        # 2. Run test.
        _result = _sh_measure.is_base_measure()

        # 3. Verify expectations.
        assert _result == expected_result
