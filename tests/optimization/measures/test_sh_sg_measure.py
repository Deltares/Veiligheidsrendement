import pytest

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure


class TestShSgMeasure:
    def _create_sh_sg_measure(
        self, measure_type: MeasureTypeEnum, combinable_type: CombinableTypeEnum
    ) -> ShSgMeasure:
        _measure = ShSgMeasure(
            measure_result_id=42,
            measure_type=measure_type,
            combine_type=combinable_type,
            dberm=0.1,
            dcrest=0.5,
        )
        return _measure

    def test_create_sh_sg_measure(self):
        # 1. Define input
        _measure_type = MeasureTypeEnum.SOIL_REINFORCEMENT
        _combine_type = CombinableTypeEnum.COMBINABLE

        # 2. Run test
        _measure = self._create_sh_sg_measure(_measure_type, _combine_type)

        # 3. Verify expectations
        assert isinstance(_measure, ShSgMeasure)
        assert isinstance(_measure, MeasureAsInputProtocol)
        assert _measure.measure_type == _measure_type
        assert _measure.combine_type == _combine_type
        assert _measure.measure_result_id == 42
        assert _measure.dberm == pytest.approx(0.1)
        assert _measure.dcrest == pytest.approx(0.5)
        assert _measure.cost == 0.0
        assert _measure.year == 0
        assert _measure.discount_rate == 0
        assert _measure.mechanism_year_collection.probabilities == []
        assert _measure.start_cost == 0.0
