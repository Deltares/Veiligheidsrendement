import pytest

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.sg_measure import SgMeasure


class TestShMeasure:

    def _create_sg_measure(
        self, measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
    ) -> SgMeasure:
        _measure = SgMeasure(
            measure_type=measure_type,
            combine_type=combinable_type,
            cost=10.5,
            year=10,
            mechanism_year_collection=None,
            dcrest=0.3,
            dberm=0.1,
            lcc=20.3,
        )
        return _measure

    def test_create_sg_measure(self):
        # 1. Define input
        _measure_type = MeasureTypeEnum.SOIL_REINFORCEMENT
        _combine_type = CombinableTypeEnum.COMBINABLE

        # 2. Run test
        _measure = self._create_sg_measure(_measure_type, _combine_type)

        # 3. Verify expectations
        assert _measure.measure_type == _measure_type
        assert _measure.combine_type == _combine_type
        assert _measure.cost == 10.5
        assert _measure.year == 10
        assert _measure.lcc == 20.3
        assert _measure.mechanism_year_collection is None
        assert _measure.dcrest == 0.3
        assert _measure.dberm == 0.1

    def test_get_allowed_mechanism(self):
        # 1./2. Define input & Run test
        _expected_mechanisms = [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING]
        _allowed_mechanisms = SgMeasure.get_allowed_mechanisms()

        # 3. Verify expectations
        assert all(x == y for x, y in zip(_expected_mechanisms, _allowed_mechanisms))

    @pytest.mark.parametrize(
        "mechanism, expected",
        [
            pytest.param(MechanismEnum.PIPING, True, id="VALID PIPING"),
            pytest.param(MechanismEnum.OVERFLOW, False, id="INVALID OVERFLOW"),
        ],
    )
    def test_is_mechanism_allowed(self, mechanism: MechanismEnum, expected: bool):
        # 1./2. Define input & Run test
        _results = SgMeasure.is_mechanism_allowed(mechanism)

        # 3. Verify expectations
        assert _results == expected

    def test_get_allowed_measure_combination(self):
        # 1./2. Define input & Run test
        _allowed_combinations = SgMeasure.get_allowed_measure_combinations()

        # 3. Verify expectations
        assert isinstance(_allowed_combinations, dict)
        assert _allowed_combinations is not None
