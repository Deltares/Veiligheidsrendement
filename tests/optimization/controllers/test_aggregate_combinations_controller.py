from __future__ import annotations

from typing import Callable, Iterator

import pytest

from tests.optimization.conftest import OverridenSgMeasure, OverridenShMeasure
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.controllers.aggregate_combinations_controller import (
    AggregateCombinationsController,
)
from vrtool.optimization.controllers.combine_measures_controller import (
    CombineMeasuresController,
)
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measures.sg_combined_measure import (
    SgCombinedMeasure,
)
from vrtool.optimization.measures.combined_measures.sh_combined_measure import (
    ShCombinedMeasure,
)
from vrtool.optimization.measures.combined_measures.shsg_combined_measure import (
    ShSgCombinedMeasure,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure


def _make_sh_measure(
    measure_type: MeasureTypeEnum, measure_result_id: int, year: int, cost: float
) -> OverridenShMeasure:
    return OverridenShMeasure(
        measure_type=measure_type,
        measure_result_id=measure_result_id,
        year=year,
        cost=cost,
    )


def _make_sg_measure(
    measure_type: MeasureTypeEnum, measure_result_id: int, year: int, cost: float
) -> OverridenSgMeasure:
    return OverridenSgMeasure(
        measure_type=measure_type,
        measure_result_id=measure_result_id,
        year=year,
        cost=cost,
    )


class TestAggregateCombinationsController:
    @pytest.fixture(name="valid_section_as_input")
    def get_section_as_input(self) -> Iterator[SectionAsInput]:
        yield SectionAsInput(
            section_name="dummy_section",
            traject_name="DummyTraject",
            flood_damage=4.2,
            measures=[],
        )

    @pytest.mark.parametrize(
        "matching_measure_type, expected_lcc, include_secondary_measure",
        [
            pytest.param(
                MeasureTypeEnum.SOIL_REINFORCEMENT,
                300,
                True,
                id=f"LCC=300 when {MeasureTypeEnum.SOIL_REINFORCEMENT.legacy_name} with initial measures and secondary measures",
            ),
            pytest.param(
                MeasureTypeEnum.SOIL_REINFORCEMENT,
                0,
                False,
                id=f"LCC=0 when {MeasureTypeEnum.SOIL_REINFORCEMENT.legacy_name} with initial measures and no secondary measures",
            ),
        ]
        + [
            pytest.param(
                _measure_type,
                450,
                True,
                id=f"LCC=450 when {_measure_type.legacy_name} with initial measures",
            )
            for _measure_type in MeasureTypeEnum
            if _measure_type != MeasureTypeEnum.SOIL_REINFORCEMENT
        ],
    )
    def test_aggregate_for_matching_year_and_type(
        self,
        valid_section_as_input: SectionAsInput,
        matching_measure_type: MeasureTypeEnum,
        include_secondary_measure: bool,
        expected_lcc: float,
    ):
        # 1. Define input

        if include_secondary_measure:
            _sh_combination = ShCombinedMeasure(
                primary=_make_sh_measure(
                    matching_measure_type,
                    1,
                    0,
                    100,
                ),
                secondary=_make_sg_measure(
                    MeasureTypeEnum.REVETMENT,
                    2,
                    0,
                    200,
                ),
                mechanism_year_collection=None,
            )

            _sg_combination = SgCombinedMeasure(
                primary=_make_sg_measure(
                    matching_measure_type,
                    1,
                    0,
                    50,
                ),
                secondary=_make_sh_measure(
                    MeasureTypeEnum.VERTICAL_PIPING_SOLUTION,
                    3,
                    0,
                    100,
                ),
                mechanism_year_collection=None,
            )
        else:
            _sh_combination = ShCombinedMeasure(
                primary=_make_sh_measure(
                    matching_measure_type,
                    1,
                    0,
                    100,
                ),
                secondary=None,
                mechanism_year_collection=None,
            )

            _sg_combination = SgCombinedMeasure(
                primary=_make_sg_measure(
                    matching_measure_type,
                    1,
                    0,
                    50,
                ),
                secondary=None,
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
        assert _aggr_meas_comb.sh_combination == _sh_combination
        assert _aggr_meas_comb.sg_combination == _sg_combination
        # Sh and Sg are initial, Sh combination is  SoilReinforcement
        assert _aggr_meas_comb.lcc == expected_lcc

    @pytest.fixture(name="sh_combination")
    def get_sh_combination(self, request: pytest.FixtureRequest) -> ShCombinedMeasure:
        return ShCombinedMeasure(
            mechanism_year_collection=None,
            primary=request.param[0],
            secondary=request.param[1],
        )

    @pytest.fixture(name="sg_combination")
    def get_sg_combination(self, request: pytest.FixtureRequest) -> SgCombinedMeasure:
        return SgCombinedMeasure(
            mechanism_year_collection=None,
            primary=request.param[0],
            secondary=request.param[1],
        )

    @pytest.mark.parametrize(
        "sh_combination, sg_combination",
        [
            pytest.param(
                [
                    _make_sh_measure(MeasureTypeEnum.SOIL_REINFORCEMENT, 1, 0, 100),
                    _make_sg_measure(MeasureTypeEnum.REVETMENT, 2, 0, 200),
                ],
                [
                    _make_sh_measure(MeasureTypeEnum.SOIL_REINFORCEMENT, 3, 20, 50),
                    _make_sg_measure(
                        MeasureTypeEnum.VERTICAL_PIPING_SOLUTION, 4, 0, 100
                    ),
                ],
                id="Non-matching year",
            ),
            pytest.param(
                [
                    _make_sh_measure(MeasureTypeEnum.DIAPHRAGM_WALL, 1, 0, 100),
                    _make_sg_measure(MeasureTypeEnum.REVETMENT, 2, 0, 200),
                ],
                [
                    _make_sg_measure(MeasureTypeEnum.SOIL_REINFORCEMENT, 3, 0, 50),
                    _make_sh_measure(
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
        sh_combination: ShCombinedMeasure,
        sg_combination: SgCombinedMeasure,
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
        _sh_combination = ShCombinedMeasure(
            mechanism_year_collection=None,
            primary=_make_sh_measure(
                MeasureTypeEnum.DIAPHRAGM_WALL,
                1,
                0,
                100,
            ),
            secondary=None,
        )
        _sg_combination = SgCombinedMeasure(
            mechanism_year_collection=None,
            primary=_make_sg_measure(
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
        _sh_combination = ShCombinedMeasure(
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
        _sg_combination = SgCombinedMeasure(
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
        _sh_combination = ShCombinedMeasure(
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
        _sg_combination = SgCombinedMeasure(
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
        _sh_combination = ShCombinedMeasure(
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
        _sg_combination = SgCombinedMeasure(
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


class TestCostComputation:
    """
    (Integration) Tests to wrap up the validation of cost computations (mostly lcc)
    """

    @pytest.fixture(name="measure_as_input_base_dict")
    def _get_measure_as_input_base_dict_fixture(self) -> Iterator[dict]:
        yield dict(
            measure_type=MeasureTypeEnum.INVALID,
            combine_type=CombinableTypeEnum.FULL,
            measure_result_id=-1,
            cost=float("nan"),
            base_cost=float("nan"),
            discount_rate=1,
            year=0,
            mechanism_year_collection=MechanismPerYearProbabilityCollection([]),
            l_stab_screen=float("nan"),
        )

    @pytest.fixture(name="create_sh_measure")
    def _get_sh_measure_factory_fixture(
        self, measure_as_input_base_dict: dict
    ) -> Iterator[Callable[[dict], ShMeasure]]:
        def create_sh_measure(sh_measure_dict: dict) -> ShMeasure:
            _base_dict = measure_as_input_base_dict | dict(
                beta_target=float("nan"),
                transition_level=float("nan"),
                dcrest=float("nan"),
            )
            return ShMeasure(**(_base_dict | sh_measure_dict))

        yield create_sh_measure

    @pytest.fixture(name="create_sg_measure")
    def _get_sg_measure_factory_fixture(
        self, measure_as_input_base_dict: dict
    ) -> Iterator[Callable[[dict], SgMeasure]]:
        def create_sg_measure(sg_measure_dict: dict) -> SgMeasure:
            _base_dict = measure_as_input_base_dict | dict(
                dberm=float("nan"),
            )
            return SgMeasure(**(_base_dict | sg_measure_dict))

        yield create_sg_measure

    @pytest.fixture(name="cost_computation_section_as_input")
    def _get_cost_computat_section_as_input_fixture(
        self,
        create_sh_measure: Callable[[dict], ShMeasure],
        create_sg_measure: Callable[[dict], SgMeasure],
    ) -> Iterator[SectionAsInput]:
        """
        VRTOOL-521 example.
        """
        _diaphragm_wall_measure_dict = dict(
            measure_type=MeasureTypeEnum.DIAPHRAGM_WALL,
            combine_type=CombinableTypeEnum.FULL,
            measure_result_id=218,
            cost=13573430,
            base_cost=13573430,
        )
        _stability_screen_measure_dict_list = [
            dict(
                measure_type=MeasureTypeEnum.STABILITY_SCREEN,
                combine_type=CombinableTypeEnum.FULL,
                measure_result_id=219,
                cost=2739300,
                base_cost=2739300,
                l_stab_screen=3,
            ),
            dict(
                measure_type=MeasureTypeEnum.STABILITY_SCREEN,
                combine_type=CombinableTypeEnum.FULL,
                measure_result_id=220,
                cost=3561090,
                base_cost=3561090,
                l_stab_screen=6,
            ),
        ]

        _sh_measures_dicts = [
            dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                combine_type=CombinableTypeEnum.COMBINABLE,
                measure_result_id=1,
                cost=100218.68,
                base_cost=100218.68,
                dcrest=0,
            ),
            dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                combine_type=CombinableTypeEnum.COMBINABLE,
                measure_result_id=9,
                cost=791920.662559160,
                base_cost=100218.68,
                dcrest=0.25,
            ),
            *_stability_screen_measure_dict_list,
            _diaphragm_wall_measure_dict,
            dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                combine_type=CombinableTypeEnum.FULL,
                measure_result_id=73,
                cost=2839518.68,
                base_cost=2839518.68,
                dcrest=0,
                l_stab_screen=3,
            ),
            dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                combine_type=CombinableTypeEnum.FULL,
                measure_result_id=74,
                cost=3661308.68,
                base_cost=3661308.68,
                dcrest=0,
                l_stab_screen=6,
            ),
            dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                combine_type=CombinableTypeEnum.FULL,
                measure_result_id=89,
                cost=3531220.66255916,
                base_cost=2839518.68,
                dcrest=0.25,
                l_stab_screen=3,
            ),
            dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                combine_type=CombinableTypeEnum.FULL,
                measure_result_id=90,
                cost=4353010.66255916,
                base_cost=3661308.68,
                dcrest=0.25,
                l_stab_screen=6,
            ),
        ]

        _sg_measures_dicts = [
            dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                combine_type=CombinableTypeEnum.COMBINABLE,
                measure_result_id=1,
                cost=100218.68,
                base_cost=100218.68,
                dberm=0,
            ),
            dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                combine_type=CombinableTypeEnum.COMBINABLE,
                measure_result_id=2,
                cost=152706.05,
                base_cost=100218.68,
                dberm=5,
            ),
            *_stability_screen_measure_dict_list,
            _diaphragm_wall_measure_dict,
            dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                combine_type=CombinableTypeEnum.FULL,
                measure_result_id=73,
                cost=2839518.68,
                base_cost=2839518.68,
                dberm=0,
                l_stab_screen=3,
            ),
            dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                combine_type=CombinableTypeEnum.FULL,
                measure_result_id=74,
                cost=3661308.68,
                base_cost=3661308.68,
                dberm=0,
                l_stab_screen=6,
            ),
            dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                combine_type=CombinableTypeEnum.FULL,
                measure_result_id=75,
                cost=2892006.05,
                base_cost=2839518.68,
                dberm=5,
                l_stab_screen=3,
            ),
            dict(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                combine_type=CombinableTypeEnum.FULL,
                measure_result_id=76,
                cost=3713796.05,
                base_cost=3661308.68,
                dberm=5,
                l_stab_screen=6,
            ),
            # SOIL_REINFORCEMENT + VZG (ID=10+217)
            dict(
                measure_type=MeasureTypeEnum.VERTICAL_PIPING_SOLUTION,
                combine_type=CombinableTypeEnum.PARTIAL,
                measure_result_id=217,
                cost=2064400,
                base_cost=100218.68,
            ),
        ]

        _sh_measures = list(map(create_sh_measure, _sh_measures_dicts))
        _sg_measures = list(map(create_sg_measure, _sg_measures_dicts))
        _sh_sg_measures = [
            ShSgMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
                combine_type=CombinableTypeEnum.COMBINABLE,
                measure_result_id=10,
                cost=844408.032559161,
                base_cost=844408.032559161,
                l_stab_screen=float("nan"),
                dcrest=0.25,
                dberm=5,
            ),
            ShSgMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                combine_type=CombinableTypeEnum.COMBINABLE,
                measure_result_id=91,
                cost=3583708.03255916,
                base_cost=3583708.03255916,
                dcrest=0.25,
                dberm=5,
                l_stab_screen=3,
            ),
            ShSgMeasure(
                measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN,
                combine_type=CombinableTypeEnum.COMBINABLE,
                measure_result_id=92,
                cost=4405498.03255916,
                base_cost=4405498.03255916,
                dcrest=0.25,
                dberm=5,
                l_stab_screen=6,
            ),
        ]
        yield SectionAsInput(
            section_name="River case VRTOOL-521",
            traject_name="test",
            flood_damage=float("nan"),
            measures=_sh_measures + _sg_measures + _sh_sg_measures,
        )

    def test_given_river_sh_sg_measures_gets_expected_lcc_values(
        self, cost_computation_section_as_input: SectionAsInput
    ):
        # 1. Define test data.
        assert isinstance(cost_computation_section_as_input, SectionAsInput)

        # 2. Run test (create combinations and aggregations).
        cost_computation_section_as_input.combined_measures = CombineMeasuresController(
            cost_computation_section_as_input
        ).combine()
        cost_computation_section_as_input.aggregated_measure_combinations = (
            AggregateCombinationsController(
                cost_computation_section_as_input
            ).aggregate()
        )

        # 3. Verify expectations.
        assert (
            len(cost_computation_section_as_input.aggregated_measure_combinations) == 19
        )

        def validate_aggregated_single_combination_lcc(
            measure_result_id: int, expected_lcc: float
        ) -> None:
            """
            Validates the aggregated LCC value when NO secondary measure
            for `sg_combination` is present.
            """
            _aggregation = next(
                (
                    am
                    for am in cost_computation_section_as_input.aggregated_measure_combinations
                    if am.sg_combination.secondary is None
                    and am.measure_result_id == measure_result_id
                ),
                None,
            )
            if _aggregation is None:
                pytest.fail(
                    f"No aggregation created for Measure result id {measure_result_id}"
                )
            assert _aggregation.lcc == pytest.approx(expected_lcc)

        # Aggregations WITHOUT secondary measures
        # Soil reinforcement aggregations
        validate_aggregated_single_combination_lcc(1, 0)
        validate_aggregated_single_combination_lcc(2, 152706.05)
        validate_aggregated_single_combination_lcc(9, 791920.662559159)
        validate_aggregated_single_combination_lcc(10, 844408.032559159)

        # Stability screen aggregations
        validate_aggregated_single_combination_lcc(219, 2739300)
        validate_aggregated_single_combination_lcc(220, 3561090)

        # Diaphragm wall aggregations
        validate_aggregated_single_combination_lcc(218, 13573430)

        # Soil reinforcement with stability screen aggregations
        validate_aggregated_single_combination_lcc(73, 2839518.68)
        validate_aggregated_single_combination_lcc(75, 2892006.05)
        validate_aggregated_single_combination_lcc(74, 3661308.68)
        validate_aggregated_single_combination_lcc(76, 3713796.05)
        validate_aggregated_single_combination_lcc(89, 3531220.66255915)
        validate_aggregated_single_combination_lcc(91, 3583708.03255915)
        validate_aggregated_single_combination_lcc(90, 4353010.66255915)
        validate_aggregated_single_combination_lcc(92, 4405498.03255915)

        # Aggregations WITH secondary measures
        def validate_aggregated_multiple_combination_lcc(
            sh_measure_primary_id: int,
            sg_measure_secondary_id: int,
            expected_lcc: float,
        ) -> None:
            """
            Gets the LCC of an aggregated measure whose `measure_result_id` primary
            and secondary measures match the provided values.
            """
            _aggregation = next(
                (
                    am
                    for am in cost_computation_section_as_input.aggregated_measure_combinations
                    if am.sg_combination.secondary is not None
                    and am.measure_result_id == sh_measure_primary_id
                    and am.sg_combination.secondary.measure_result_id
                    == sg_measure_secondary_id
                ),
                None,
            )
            if _aggregation is None:
                pytest.fail(
                    f"No aggregation found for 'MeasureResultId'={sh_measure_primary_id} and 'sg_combination.secondary'={sg_measure_secondary_id}"
                )
            assert _aggregation.lcc == pytest.approx(expected_lcc)

        # Soil reinforcement with vertical piping solution
        validate_aggregated_multiple_combination_lcc(10, 217, 2908808.03255916)

    def test_given_aggregations_with_shsg_combined_measures_get_their_value(
        self, cost_computation_section_as_input: SectionAsInput
    ):
        # 1. Define test data.
        assert isinstance(cost_computation_section_as_input, SectionAsInput)

        # 2. Run test.
        cost_computation_section_as_input.combined_measures = CombineMeasuresController(
            cost_computation_section_as_input
        ).combine()

        _aggregated_measure_combinations = AggregateCombinationsController(
            cost_computation_section_as_input
        ).aggregate()

        # 3. Verify expectations.
        assert any(_aggregated_measure_combinations)
        assert all(
            isinstance(_amc, AggregatedMeasureCombination)
            for _amc in _aggregated_measure_combinations
        )

        # Get the subset we actually want to test.
        _with_shsg_combinations = list(
            filter(
                lambda x: isinstance(x.shsg_combination, ShSgCombinedMeasure),
                _aggregated_measure_combinations,
            )
        )
        assert any(_with_shsg_combinations)

        for _amc_with_shsg in _with_shsg_combinations:
            assert _amc_with_shsg.lcc == pytest.approx(
                _amc_with_shsg.shsg_combination.lcc
            )
