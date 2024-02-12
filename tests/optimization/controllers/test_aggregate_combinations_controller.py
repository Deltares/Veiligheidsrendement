from __future__ import annotations

from dataclasses import dataclass

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.controllers.aggregate_combinations_controller import (
    AggregateCombinationsController,
)
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)


@dataclass
class MockCombinedMeasure:
    primary: MockMeasure
    secondary: MockMeasure
    lcc: float


@dataclass
class MockMeasure(MeasureAsInputProtocol):
    measure_type: MeasureTypeEnum
    year: int
    lcc: float


class TestAggregateCombinationsController:
    def test_aggregate_for_matching_year_and_type(self):
        # 1. Define input
        _sh_combination = MockCombinedMeasure(
            MockMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 0, 100),
            MockMeasure(MeasureTypeEnum.REVETMENT, 0, 200),
            300,
        )
        _sg_combination = MockCombinedMeasure(
            MockMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 0, 50),
            MockMeasure(MeasureTypeEnum.VERTICAL_GEOTEXTILE, 0, 100),
            150,
        )

        # 2. Run test
        _aggr_meas_comb = AggregateCombinationsController._create_aggregates(
            [_sh_combination], [_sg_combination]
        )

        # 3. Verify expectations
        assert len(_aggr_meas_comb) == 1
        assert _aggr_meas_comb[0].lcc == 450

    def test_aggregate_for_non_matching_year(self):
        # 1. Define input
        _sh_combination = MockCombinedMeasure(
            MockMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 0, 100),
            MockMeasure(MeasureTypeEnum.REVETMENT, 0, 200),
            300,
        )
        _sg_combination = MockCombinedMeasure(
            MockMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 20, 50),
            MockMeasure(MeasureTypeEnum.VERTICAL_GEOTEXTILE, 0, 100),
            150,
        )

        # 2. Run test
        _aggr_meas_comb = AggregateCombinationsController._create_aggregates(
            [_sh_combination], [_sg_combination]
        )

        # 3. Verify expectations
        assert len(_aggr_meas_comb) == 0

    def test_aggregate_for_non_matching_type(self):
        # 1. Define input
        _sh_combination = MockCombinedMeasure(
            MockMeasure(MeasureTypeEnum.DIAPHRAGM_WALL, 0, 100),
            MockMeasure(MeasureTypeEnum.REVETMENT, 0, 200),
            300,
        )
        _sg_combination = MockCombinedMeasure(
            MockMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 0, 50),
            MockMeasure(MeasureTypeEnum.VERTICAL_GEOTEXTILE, 0, 100),
            150,
        )

        # 2. Run test
        _aggr_meas_comb = AggregateCombinationsController._create_aggregates(
            [_sh_combination], [_sg_combination]
        )

        # 3. Verify expectations
        assert len(_aggr_meas_comb) == 0
