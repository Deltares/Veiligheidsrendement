import pytest

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.optimization.strategy_input.strategy_input import StrategyInput


class TestStrategyInput:

    def test_initialize_with_required_properties(self):
        # 1. Define test tata.
        _required_args = dict(design_method="dummy_method")

        # 2. Run test.
        _strategy_input = StrategyInput(**_required_args)

        # 3. Verify final expectations.
        assert isinstance(_strategy_input, StrategyInput)
        assert _strategy_input.design_method == _required_args["design_method"]

    def test_optimization_input(self):
        # 1. Define input
        _design_method = "dummy_method"

        # Measures
        # - Sh soil year 0/20
        _mech_yr_coll_sh_soil_0 = MechanismPerYearProbabilityCollection(
            [
                MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.001),
                MechanismPerYear(MechanismEnum.OVERFLOW, 50, 0.0011),
                MechanismPerYear(MechanismEnum.REVETMENT, 0, 0.003),
                MechanismPerYear(MechanismEnum.REVETMENT, 50, 0.0031),
            ]
        )
        _sh_measure_soil_0_0 = ShMeasure(
            measure_result_id=0,
            measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
            combine_type=CombinableTypeEnum.COMBINABLE,
            cost=193369,
            discount_rate=0.03,
            year=0,
            mechanism_year_collection=_mech_yr_coll_sh_soil_0,
            dcrest=0,
            beta_target=-999,
            transition_level=-999,
        )
        _sh_measure_soil_20_0 = ShMeasure(
            measure_result_id=0,
            measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
            combine_type=CombinableTypeEnum.COMBINABLE,
            cost=193369,
            discount_rate=0.03,
            year=20,
            mechanism_year_collection=_mech_yr_coll_sh_soil_0,
            dcrest=0,
            beta_target=-999,
            transition_level=-999,
        )
        # - Sh Diaphgragm wall year 0
        _mech_yr_coll_sh_diaphragm_0 = MechanismPerYearProbabilityCollection(
            [
                MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.001),
                MechanismPerYear(MechanismEnum.OVERFLOW, 50, 0.0011),
                MechanismPerYear(MechanismEnum.REVETMENT, 0, 0.002),
                MechanismPerYear(MechanismEnum.REVETMENT, 50, 0.0021),
            ]
        )
        _sh_measure_diaphragm_0 = ShMeasure(
            measure_result_id=0,
            measure_type=MeasureTypeEnum.DIAPHRAGM_WALL,
            combine_type=CombinableTypeEnum.FULL,
            cost=234567,
            discount_rate=0.03,
            year=0,
            mechanism_year_collection=_mech_yr_coll_sh_diaphragm_0,
            dcrest=-999,
            beta_target=-999,
            transition_level=-999,
        )
        # - Sh Revetment year 0 transition 3.84
        _mech_yr_coll_sh_revetment_0_384 = MechanismPerYearProbabilityCollection(
            [
                MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.001),
                MechanismPerYear(MechanismEnum.OVERFLOW, 50, 0.0011),
                MechanismPerYear(MechanismEnum.REVETMENT, 0, 0.002),
                MechanismPerYear(MechanismEnum.REVETMENT, 50, 0.0021),
            ]
        )
        _sh_measure_revetment_0_384 = ShMeasure(
            measure_result_id=0,
            measure_type=MeasureTypeEnum.REVETMENT,
            combine_type=CombinableTypeEnum.REVETMENT,
            cost=123456,
            discount_rate=0.03,
            year=0,
            mechanism_year_collection=_mech_yr_coll_sh_revetment_0_384,
            dcrest=-999,
            beta_target=4.648,
            transition_level=3.84,
        )
        # - Sh Revetment year 0 transition 4.84
        _mech_yr_coll_sh_revetment_0_484 = MechanismPerYearProbabilityCollection(
            [
                MechanismPerYear(MechanismEnum.OVERFLOW, 0, 0.001),
                MechanismPerYear(MechanismEnum.OVERFLOW, 50, 0.0011),
                MechanismPerYear(MechanismEnum.REVETMENT, 0, 0.002),
                MechanismPerYear(MechanismEnum.REVETMENT, 50, 0.0021),
            ]
        )
        _sh_measure_revetment_0_484 = ShMeasure(
            measure_result_id=0,
            measure_type=MeasureTypeEnum.REVETMENT,
            combine_type=CombinableTypeEnum.REVETMENT,
            cost=223456,
            discount_rate=0.03,
            year=0,
            mechanism_year_collection=_mech_yr_coll_sh_revetment_0_484,
            dcrest=-999,
            beta_target=4.648,
            transition_level=4.84,
        )
        # - Sg soil year 0/20
        _mech_yr_coll_sg_soil_0 = MechanismPerYearProbabilityCollection(
            [
                MechanismPerYear(MechanismEnum.STABILITY_INNER, 0, 0.001),
                MechanismPerYear(MechanismEnum.STABILITY_INNER, 50, 0.0011),
                MechanismPerYear(MechanismEnum.PIPING, 0, 0.003),
                MechanismPerYear(MechanismEnum.PIPING, 50, 0.0031),
            ]
        )
        _sg_measure_soil_0_0 = SgMeasure(
            measure_result_id=0,
            measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
            combine_type=CombinableTypeEnum.COMBINABLE,
            cost=193369,
            discount_rate=0.03,
            year=0,
            mechanism_year_collection=_mech_yr_coll_sg_soil_0,
            dberm=0,
        )
        _sg_measure_soil_20_0 = SgMeasure(
            measure_result_id=0,
            measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
            combine_type=CombinableTypeEnum.COMBINABLE,
            cost=193369,
            discount_rate=0.03,
            year=20,
            mechanism_year_collection=_mech_yr_coll_sg_soil_0,
            dberm=0,
        )
        # - Sg VZG year 0
        _mech_yr_coll_sg_vzg_0 = MechanismPerYearProbabilityCollection(
            [
                MechanismPerYear(MechanismEnum.STABILITY_INNER, 0, 0.001),
                MechanismPerYear(MechanismEnum.STABILITY_INNER, 50, 0.0011),
                MechanismPerYear(MechanismEnum.PIPING, 0, 0.002),
                MechanismPerYear(MechanismEnum.PIPING, 50, 0.0021),
            ]
        )
        _sg_measure_vzg_0 = SgMeasure(
            measure_result_id=0,
            measure_type=MeasureTypeEnum.VERTICAL_GEOTEXTILE,
            combine_type=CombinableTypeEnum.PARTIAL,
            cost=1302200,
            discount_rate=0.03,
            year=0,
            mechanism_year_collection=_mech_yr_coll_sg_vzg_0,
            dberm=-999,
        )

        # Sections
        _initial_assessment = MechanismPerYearProbabilityCollection(
            _sh_measure_soil_0_0.mechanism_year_collection.probabilities
            + _sg_measure_soil_0_0.mechanism_year_collection.probabilities
        )
        _sections = [
            SectionAsInput(
                section_name="section_1",
                traject_name="traject_1",
                flood_damage=0,
                measures=[
                    _sh_measure_soil_0_0,
                    _sh_measure_soil_20_0,
                    _sh_measure_revetment_0_384,
                    _sh_measure_revetment_0_484,
                    _sh_measure_diaphragm_0,
                    _sg_measure_soil_0_0,
                    _sg_measure_soil_20_0,
                    _sg_measure_vzg_0,
                ],
                initial_assessment=_initial_assessment,
            )
        ]

        # Combinations
        _sh_combination_soil_0 = CombinedMeasure.from_input(
            _sh_measure_soil_0_0, None, _initial_assessment, 0
        )
        _sh_combination_soil_20 = CombinedMeasure.from_input(
            _sh_measure_soil_20_0, None, _initial_assessment, 1
        )
        _sh_combination_soil_revetment_0_384 = CombinedMeasure.from_input(
            _sh_measure_soil_0_0, _sh_measure_revetment_0_384, _initial_assessment, 2
        )
        _sh_combination_soil_revetment_20_384 = CombinedMeasure.from_input(
            _sh_measure_soil_20_0, _sh_measure_revetment_0_384, _initial_assessment, 3
        )
        _sh_combination_soil_revetment_0_484 = CombinedMeasure.from_input(
            _sh_measure_soil_0_0, _sh_measure_revetment_0_484, _initial_assessment, 4
        )
        _sh_combination_soil_revetment_20_484 = CombinedMeasure.from_input(
            _sh_measure_soil_20_0, _sh_measure_revetment_0_484, _initial_assessment, 5
        )
        _sh_combination_diaphragm_0 = CombinedMeasure.from_input(
            _sh_measure_diaphragm_0, None, _initial_assessment, 6
        )

        _sg_combination_soil_0 = CombinedMeasure.from_input(
            _sg_measure_soil_0_0, None, _initial_assessment, 0
        )
        _sg_combination_soil_20 = CombinedMeasure.from_input(
            _sg_measure_soil_20_0, None, _initial_assessment, 1
        )
        _sg_combination_soil_vzg_0 = CombinedMeasure.from_input(
            _sg_measure_soil_0_0, _sg_measure_vzg_0, _initial_assessment, 2
        )
        _sg_combination_soil_vzg_20 = CombinedMeasure.from_input(
            _sg_measure_soil_20_0, _sg_measure_vzg_0, _initial_assessment, 3
        )

        _sections[0].combined_measures = [
            _sh_combination_soil_0,
            _sh_combination_soil_20,
            _sh_combination_soil_revetment_0_384,
            _sh_combination_soil_revetment_20_384,
            _sh_combination_soil_revetment_0_484,
            _sh_combination_soil_revetment_20_484,
            _sh_combination_diaphragm_0,
            _sg_combination_soil_0,
            _sg_combination_soil_20,
            _sg_combination_soil_vzg_0,
            _sg_combination_soil_vzg_20,
        ]

        # Aggregations
        _sections[0].aggregated_measure_combinations = [
            AggregatedMeasureCombination(
                _sh_combination_soil_0, _sg_combination_soil_0, 1, 0
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_0_384, _sg_combination_soil_0, 2, 0
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_0_484, _sg_combination_soil_0, 3, 0
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_0, _sg_combination_soil_vzg_0, 4, 0
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_0_384, _sg_combination_soil_vzg_0, 5, 0
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_0_484, _sg_combination_soil_vzg_0, 6, 0
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_20, _sg_combination_soil_20, 1, 20
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_20_384, _sg_combination_soil_20, 2, 20
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_20_484, _sg_combination_soil_20, 3, 20
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_20, _sg_combination_soil_vzg_20, 4, 20
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_20_384,
                _sg_combination_soil_vzg_20,
                5,
                20,
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_20_484,
                _sg_combination_soil_vzg_20,
                6,
                20,
            ),
        ]

        # 2. Run test
        _strategy_input = StrategyInput.from_section_as_input_collection(
            _sections, _design_method
        )

        # 3. Verify expectations

        # Probabilities
        assert _strategy_input.Pf
        assert _strategy_input.Pf[MechanismEnum.OVERFLOW.name].shape == (
            1,
            8,
            50,
        )
        assert _strategy_input.Pf[MechanismEnum.REVETMENT.name].shape == (
            1,
            8,
            50,
        )
        assert _strategy_input.Pf[MechanismEnum.PIPING.name].shape == (
            1,
            5,
            50,
        )
        assert _strategy_input.Pf[MechanismEnum.STABILITY_INNER.name].shape == (
            1,
            5,
            50,
        )
        assert _strategy_input.Pf[MechanismEnum.STABILITY_INNER.name][
            0, 0, 0
        ] == pytest.approx(0.0010)

        # Cost
        assert _strategy_input.LCCOption is not None
        assert _strategy_input.LCCOption.shape == (1, 8, 5)
        assert _strategy_input.LCCOption[0, 0, 0] == pytest.approx(0.0)
        assert _strategy_input.LCCOption[0, 1, 1] == pytest.approx(386738.0)
        assert _strategy_input.LCCOption[0, 1, 2] == pytest.approx(1e99)
        assert _strategy_input.LCCOption[0, 1, 3] == pytest.approx(1688938.0)
        assert _strategy_input.LCCOption[0, 2, 2] == pytest.approx(214127.4538)
        assert _strategy_input.LCCOption[0, 3, 3] == pytest.approx(1812394.0)
        assert _strategy_input.LCCOption[0, 4, 4] == pytest.approx(1639783.4538)

        # Other structures
        assert _strategy_input.Cint_h.shape == (1, 7)
        assert _strategy_input.Cint_g.shape == (1, 4)
        assert _strategy_input.Dint.shape == (1, 7)
        assert _strategy_input.D.shape == (50,)
        assert _strategy_input.RiskGeotechnical.shape == (1, 5, 50)
        assert _strategy_input.RiskOverflow.shape == (1, 8, 50)
        assert _strategy_input.RiskRevetment.shape == (1, 8, 50)
