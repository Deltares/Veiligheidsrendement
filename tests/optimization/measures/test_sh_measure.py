from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.sh_measure import ShMeasure


class TestShMeasure:

    def _create_sh_measure(
        self, measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
    ) -> ShMeasure:
        _measure = ShMeasure(
            measure_type=measure_type,
            combine_type=combinable_type,
            cost=10.5,
            year=10,
            mechanism_year_collection=None,
            beta_target=1.1,
            transition_level=0.5,
            dcrest=0.1,
        )
        _measure.lcc = 20.3
        return _measure

    def test_create_sh_measure(self):
        # 1. Define input
        _measure_type = MeasureTypeEnum.SOIL_REINFORCEMENT
        _combine_type = CombinableTypeEnum.COMBINABLE

        # 2. Run test
        _measure = self._create_sh_measure(_measure_type, _combine_type)

        # 3. Verify expectations
        assert _measure.measure_type == _measure_type
        assert _measure.combine_type == _combine_type
        assert _measure.cost == 10.5
        assert _measure.year == 10
        assert _measure.lcc == 20.3
        assert _measure.mechanism_year_collection is None
        assert _measure.beta_target == 1.1
        assert _measure.transition_level == 0.5
        assert _measure.dcrest == 0.1

    def test_sh_measure_with_type_sets_lcc_0(self):
        # 1. Define input
        _measure_type = MeasureTypeEnum.DIAPHRAGM_WALL
        _combine_type = CombinableTypeEnum.FULL
        _measure = self._create_sh_measure(_measure_type, _combine_type)

        # 2. Run test
        _lcc_after = _measure.lcc

        # 3. Verify expectations
        assert _lcc_after == 0
