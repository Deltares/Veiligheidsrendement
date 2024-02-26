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
class MockSectionAsInput:
    @property
    def sg_sh_measures(self) -> list[MeasureAsInputProtocol]:
        return [
            MockMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=3,
                year=0,
                cost=100,
                dberm=1.0,
                dcrest=0.5,
            )
        ]


@dataclass
class MockCombinedMeasure:
    primary: MockMeasure
    secondary: MockMeasure

    @property
    def lcc(self) -> float:
        if self.secondary is not None:
            return self.primary.lcc + self.secondary.lcc
        return self.primary.lcc


@dataclass
class MockMeasure(MeasureAsInputProtocol):
    measure_type: MeasureTypeEnum
    measure_result_id: int
    year: int
    cost: float
    dcrest: float = 0
    dberm: float = 0

    @property
    def lcc(self) -> float:
        return self.cost


class TestAggregateCombinationsController:
    def test_aggregate_for_matching_year_and_type(self):
        # 1. Define input
        _sh_combination = MockCombinedMeasure(
            MockMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 1, 0, 100),
            MockMeasure(MeasureTypeEnum.REVETMENT, 2, 0, 200),
        )
        _sg_combination = MockCombinedMeasure(
            MockMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 3, 0, 50),
            MockMeasure(MeasureTypeEnum.VERTICAL_GEOTEXTILE, 4, 0, 100),
        )

        # 2. Run test
        _aggr_meas_comb = AggregateCombinationsController(None)._create_aggregates(
            [_sh_combination], [_sg_combination]
        )

        # 3. Verify expectations
        assert len(_aggr_meas_comb) == 1
        assert _aggr_meas_comb[0].lcc == 450
        assert _aggr_meas_comb[0].sh_combination == _sh_combination
        assert _aggr_meas_comb[0].sg_combination == _sg_combination

    def test_aggregate_for_non_matching_year(self):
        # 1. Define input
        _sh_combination = MockCombinedMeasure(
            MockMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 1, 0, 100),
            MockMeasure(MeasureTypeEnum.REVETMENT, 2, 0, 200),
        )
        _sg_combination = MockCombinedMeasure(
            MockMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 3, 20, 50),
            MockMeasure(MeasureTypeEnum.VERTICAL_GEOTEXTILE, 4, 0, 100),
        )

        # 2. Run test
        _aggr_meas_comb = AggregateCombinationsController(None)._create_aggregates(
            [_sh_combination], [_sg_combination]
        )

        # 3. Verify expectations
        assert len(_aggr_meas_comb) == 0

    def test_aggregate_for_non_matching_type(self):
        # 1. Define input
        _sh_combination = MockCombinedMeasure(
            MockMeasure(MeasureTypeEnum.DIAPHRAGM_WALL, 1, 0, 100),
            MockMeasure(MeasureTypeEnum.REVETMENT, 2, 0, 200),
        )
        _sg_combination = MockCombinedMeasure(
            MockMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 3, 0, 50),
            MockMeasure(MeasureTypeEnum.VERTICAL_GEOTEXTILE, 4, 0, 100),
        )

        # 2. Run test
        _aggr_meas_comb = AggregateCombinationsController(None)._create_aggregates(
            [_sh_combination], [_sg_combination]
        )

        # 3. Verify expectations
        assert len(_aggr_meas_comb) == 0

    def test_aggregated_measure_id_for_matching_measure_result_id(self):
        # 1. Define input
        _sh_combination = MockCombinedMeasure(
            MockMeasure(
                measure_type=MeasureTypeEnum.DIAPHRAGM_WALL,
                measure_result_id=1,
                year=0,
                cost=100,
            ),
            None,
        )
        _sg_combination = MockCombinedMeasure(
            MockMeasure(
                measure_type=MeasureTypeEnum.DIAPHRAGM_WALL,
                measure_result_id=1,
                year=0,
                cost=50,
            ),
            None,
        )

        # 2. Run test
        _aggr_meas_comb = AggregateCombinationsController(None)._create_aggregates(
            [_sh_combination], [_sg_combination]
        )

        # 3. Verify expectations
        assert len(_aggr_meas_comb) == 1
        assert _aggr_meas_comb[0].measure_result_id == 1

    def test_aggregated_measure_id_returns_sg_for_sh_measure_with_dcrest_0(self):
        # 1. Define input
        _sh_combination = MockCombinedMeasure(
            MockMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=1,
                year=0,
                cost=100,
                dcrest=0.0,
            ),
            None,
        )
        _sg_combination = MockCombinedMeasure(
            MockMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=2,
                year=0,
                cost=50,
                dberm=0.5,
            ),
            None,
        )

        # 2. Run test
        _aggr_meas_comb = AggregateCombinationsController(None)._create_aggregates(
            [_sh_combination], [_sg_combination]
        )

        # 3. Verify expectations
        assert len(_aggr_meas_comb) == 1
        assert _aggr_meas_comb[0].measure_result_id == 2

    def test_aggregated_measure_id_returns_sh_for_sg_measure_with_dberm_0(self):
        # 1. Define input
        _sh_combination = MockCombinedMeasure(
            MockMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=1,
                year=0,
                cost=100,
                dcrest=0.5,
            ),
            None,
        )
        _sg_combination = MockCombinedMeasure(
            MockMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=2,
                year=0,
                cost=50,
                dberm=0.0,
            ),
            None,
        )

        # 2. Run test
        _aggr_meas_comb = AggregateCombinationsController(None)._create_aggregates(
            [_sh_combination], [_sg_combination]
        )

        # 3. Verify expectations
        assert len(_aggr_meas_comb) == 1
        assert _aggr_meas_comb[0].measure_result_id == 1

    def test_aggregated_measure_id_returns_sg_sh(self):
        # 1. Define input
        _sh_combination = MockCombinedMeasure(
            MockMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=1,
                year=0,
                cost=100,
                dcrest=0.5,
            ),
            None,
        )
        _sg_combination = MockCombinedMeasure(
            MockMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=2,
                year=0,
                cost=50,
                dberm=1.0,
            ),
            None,
        )

        # 2. Run test
        _aggr_controller = AggregateCombinationsController(None)
        _aggr_controller._section = MockSectionAsInput()
        _aggr_meas_comb = _aggr_controller._create_aggregates(
            [_sh_combination], [_sg_combination]
        )

        # 3. Verify expectations
        assert len(_aggr_meas_comb) == 1
        assert _aggr_meas_comb[0].measure_result_id == 3
