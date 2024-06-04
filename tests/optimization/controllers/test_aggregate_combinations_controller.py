from __future__ import annotations

from typing import Iterator

import pytest

from tests.optimization import OverridenSgMeasure, OverridenShMeasure
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.controllers.aggregate_combinations_controller import (
    AggregateCombinationsController,
)
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.section_as_input import SectionAsInput


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
        self,
        valid_section_as_input: SectionAsInput,
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

    def test_aggregated_measure_id_returns_without_matching_measure_result_id_raises(
        self, valid_section_as_input: SectionAsInput
    ):
        """
        This tests validates that we DO NOT support handling aggregating measures that
        do not have any "shared" measure result (VRTOOL-518).
        Usually this error would not pop up until the export of the results.
        """
        # 1. Define input
        _sh_id = 1
        _sg_id = 2
        _sh_combination = CombinedMeasure(
            mechanism_year_collection=None,
            primary=OverridenShMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                measure_result_id=_sh_id,
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
                measure_result_id=_sg_id,
                year=0,
                cost=50,
                dberm=1.0,
            ),
            secondary=None,
        )
        _expected_error = f"Geen `MeasureResult.id` gevonden tussen gecombineerd (primary) maatregelen met `MeasureResult.id`: Sh ({_sh_id}) en Sg ({_sg_id})."

        # 2. Run test
        valid_section_as_input.combined_measures.append(_sh_combination)
        valid_section_as_input.combined_measures.append(_sg_combination)

        # 2. Run test
        with pytest.raises(ValueError) as exc_err:
            AggregateCombinationsController(valid_section_as_input).aggregate()

        # 3. Verify expectations
        assert str(exc_err.value) == _expected_error
