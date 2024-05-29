from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pytest

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.controllers.aggregate_combinations_controller import (
    AggregateCombinationsController,
)
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


@dataclass
class OverridenShMeasure(ShMeasure):
    measure_type: MeasureTypeEnum
    measure_result_id: int = -1
    year: int = 0
    cost: float = float("nan")
    combine_type: CombinableTypeEnum = CombinableTypeEnum.FULL
    discount_rate: float = 0
    mechanism_year_collection: MechanismPerYearProbabilityCollection = None
    beta_target: float = float("nan")
    transition_level: float = float("nan")
    dcrest: float = 0
    l_stab_screen: float = float("nan")


@dataclass
class OverridenSgMeasure(SgMeasure):
    measure_type: MeasureTypeEnum
    measure_result_id: int = -1
    year: int = 0
    cost: float = float("nan")
    combine_type: CombinableTypeEnum = CombinableTypeEnum.FULL
    discount_rate: float = float("nan")
    mechanism_year_collection: MechanismPerYearProbabilityCollection = None
    dberm: float = float("nan")
    l_stab_screen: float = float("nan")


class TestAggregateCombinationsController:
    @pytest.fixture(name="valid_section_as_input")
    def get_section_as_input(self) -> Iterator[SectionAsInput]:
        yield SectionAsInput(
            section_name="dummy_section",
            traject_name="DummyTraject",
            flood_damage=4.2,
            measures=[],
        )

    def test_aggregate_for_matching_year_and_type(
        self, valid_section_as_input: SectionAsInput
    ):
        # 1. Define input
        _sh_combination = CombinedMeasure(
            primary=OverridenShMeasure(
                MeasureTypeEnum.SOIL_REINFORCEMENT,
                1,
                0,
                100,
            ),
            secondary=OverridenSgMeasure(
                MeasureTypeEnum.REVETMENT,
                2,
                0,
                200,
            ),
            mechanism_year_collection=None,
        )
        _sg_combination = CombinedMeasure(
            primary=OverridenSgMeasure(
                MeasureTypeEnum.SOIL_REINFORCEMENT,
                3,
                0,
                50,
            ),
            secondary=OverridenShMeasure(
                MeasureTypeEnum.VERTICAL_PIPING_SOLUTION,
                4,
                0,
                100,
            ),
            mechanism_year_collection=None,
        )
        valid_section_as_input.combined_measures.append(_sh_combination)
        valid_section_as_input.combined_measures.append(_sg_combination)

        # 2. Run test
        _created_aggregations = AggregateCombinationsController(
            valid_section_as_input
        ).aggregate()

        # 3. Verify expectations
        assert len(_created_aggregations) == 1
        _aggr_meas_comb = _created_aggregations[0]
        assert isinstance(_aggr_meas_comb, AggregatedMeasureCombination)
        assert _aggr_meas_comb.lcc == 250
        assert _aggr_meas_comb.sh_combination == _sh_combination
        assert _aggr_meas_comb.sg_combination == _sg_combination

    @pytest.fixture(name="sh_combination")
    def get_sh_combination(self, request: pytest.FixtureRequest) -> CombinedMeasure:
        return CombinedMeasure(
            mechanism_year_collection=None,
            primary=request.param[0],
            secondary=request.param[1],
        )

    @pytest.fixture(name="sg_combination")
    def get_sg_combination(self, request: pytest.FixtureRequest) -> CombinedMeasure:
        return CombinedMeasure(
            mechanism_year_collection=None,
            primary=request.param[0],
            secondary=request.param[1],
        )

    @pytest.mark.parametrize(
        "sh_combination, sg_combination",
        [
            pytest.param(
                [
                    OverridenShMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 1, 0, 100),
                    OverridenSgMeasure(MeasureTypeEnum.REVETMENT, 2, 0, 200),
                ],
                [
                    OverridenShMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 3, 20, 50),
                    OverridenSgMeasure(
                        MeasureTypeEnum.VERTICAL_PIPING_SOLUTION, 4, 0, 100
                    ),
                ],
                id="Non-matching year",
            ),
            pytest.param(
                [
                    OverridenShMeasure(MeasureTypeEnum.DIAPHRAGM_WALL, 1, 0, 100),
                    OverridenSgMeasure(MeasureTypeEnum.REVETMENT, 2, 0, 200),
                ],
                [
                    OverridenSgMeasure(MeasureTypeEnum.SOIL_REINFORCEMENT, 3, 0, 50),
                    OverridenShMeasure(
                        MeasureTypeEnum.VERTICAL_PIPING_SOLUTION, 4, 0, 100
                    ),
                ],
                id="Non-matching type",
            ),
        ],
        indirect=True,
    )
    def test_aggregate_for_non_compatible_returns_empty_list(
        self,
        sh_combination: CombinedMeasure,
        sg_combination: CombinedMeasure,
        valid_section_as_input: SectionAsInput,
    ):
        # 1. Define input
        valid_section_as_input.combined_measures.append(sh_combination)
        valid_section_as_input.combined_measures.append(sg_combination)

        # 2. Run test
        _aggr_meas_comb = AggregateCombinationsController(
            valid_section_as_input
        ).aggregate()

        # 3. Verify expectations
        assert not any(_aggr_meas_comb)

    def test_aggregated_measure_id_for_matching_measure_result_id(
        self, valid_section_as_input: SectionAsInput
    ):
        # 1. Define input
        _sh_combination = CombinedMeasure(
            mechanism_year_collection=None,
            primary=OverridenShMeasure(
                MeasureTypeEnum.DIAPHRAGM_WALL,
                1,
                0,
                100,
            ),
            secondary=None,
        )
        _sg_combination = CombinedMeasure(
            mechanism_year_collection=None,
            primary=OverridenSgMeasure(
                MeasureTypeEnum.DIAPHRAGM_WALL,
                2,
                0,
                50,
            ),
            secondary=None,
        )

        valid_section_as_input.combined_measures.append(_sh_combination)
        valid_section_as_input.combined_measures.append(_sg_combination)

        # 2. Run test
        _created_aggregations = AggregateCombinationsController(
            valid_section_as_input
        ).aggregate()

        # 3. Verify expectations
        assert len(_created_aggregations) == 1
        assert (
            _created_aggregations[0].measure_result_id
            == _sg_combination.primary.measure_result_id
        )

    def test_aggregated_measure_id_returns_sg_for_sh_measure_with_dcrest_0(
        self, valid_section_as_input: SectionAsInput
    ):
        # 1. Define input
        _sh_combination = CombinedMeasure(
            mechanism_year_collection=None,
            primary=OverridenShMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=1,
                year=0,
                cost=100,
                dcrest=0.0,
            ),
            secondary=None,
        )
        _sg_combination = CombinedMeasure(
            mechanism_year_collection=None,
            primary=OverridenSgMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=2,
                year=0,
                cost=50,
                dberm=0.5,
            ),
            secondary=None,
        )

        valid_section_as_input.combined_measures.append(_sh_combination)
        valid_section_as_input.combined_measures.append(_sg_combination)

        # 2. Run test
        _created_aggregations = AggregateCombinationsController(
            valid_section_as_input
        ).aggregate()

        # 3. Verify expectations
        assert len(_created_aggregations) == 1
        assert (
            _created_aggregations[0].measure_result_id
            == _sg_combination.primary.measure_result_id
        )

    def test_aggregated_measure_id_returns_sh_for_sg_measure_with_dberm_0(
        self, valid_section_as_input: SectionAsInput
    ):
        # 1. Define input
        _sh_combination = CombinedMeasure(
            mechanism_year_collection=None,
            primary=OverridenShMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=1,
                year=0,
                cost=100,
                dcrest=0.5,
            ),
            secondary=None,
        )
        _sg_combination = CombinedMeasure(
            mechanism_year_collection=None,
            primary=OverridenSgMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=2,
                year=0,
                cost=50,
                dberm=0.0,
            ),
            secondary=None,
        )

        valid_section_as_input.combined_measures.append(_sh_combination)
        valid_section_as_input.combined_measures.append(_sg_combination)

        # 2. Run test
        _created_aggregations = AggregateCombinationsController(
            valid_section_as_input
        ).aggregate()

        # 3. Verify expectations
        assert len(_created_aggregations) == 1
        assert (
            _created_aggregations[0].measure_result_id
            == _sh_combination.primary.measure_result_id
        )

    def test_aggregated_measure_id_returns_sh_sg(
        self, valid_section_as_input: SectionAsInput
    ):
        # 1. Define input
        _sh_combination = CombinedMeasure(
            mechanism_year_collection=None,
            primary=OverridenShMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=1,
                year=0,
                cost=100,
                dcrest=0.5,
            ),
            secondary=None,
        )
        _sg_combination = CombinedMeasure(
            mechanism_year_collection=None,
            primary=OverridenSgMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=2,
                year=0,
                cost=50,
                dberm=1.0,
            ),
            secondary=None,
        )

        # 2. Run test
        # _aggr_controller._section = MockSectionAsInput(measure_result_id=3)
        valid_section_as_input.combined_measures.append(_sh_combination)
        valid_section_as_input.combined_measures.append(_sg_combination)

        # 2. Run test
        _created_aggregations = AggregateCombinationsController(
            valid_section_as_input
        ).aggregate()

        # 3. Verify expectations
        assert len(_created_aggregations) == 1
        # VRTOOL-518
        # When no `ShSgMeasure` is present we simply return a 0.
        # this will later on give an exception when trying to export.
        assert _created_aggregations[0].measure_result_id == 0
