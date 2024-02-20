import pytest
from pandas import DataFrame as df

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.optimization.controllers.strategy_controller import StrategyController
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

SECTION1 = "Section_1"
SECTION2 = "Section_2"
TRAJECT1 = "Traject_1"


class TestStrategyController:
    def _create_valid_dike_section(self, section_name: str):
        _dike_section = DikeSection()
        _dike_section.name = section_name
        return _dike_section

    def _create_valid_dike_traject(self):
        _dike_traject = DikeTraject()
        _dike_traject.general_info = DikeTrajectInfo(traject_name=TRAJECT1)
        _section1 = self._create_valid_dike_section(SECTION1)
        _section2 = self._create_valid_dike_section(SECTION2)
        _dike_traject.sections = [_section1, _section2]
        return _dike_traject

    def _create_measures_df(self) -> df:
        return df.from_dict(
            {
                ("type", ""): [
                    "Soil reinforcement",
                    "Soil reinforcement",
                    "Soil reinforcement",
                    "Soil reinforcement",
                    "Soil reinforcement",
                    "Soil reinforcement",
                    "Soil reinforcement",
                    "Soil reinforcement",
                    "Soil reinforcement with stability screen",
                    "Soil reinforcement with stability screen",
                    "Soil reinforcement with stability screen",
                    "Soil reinforcement with stability screen",
                    "Vertical Geotextile",
                    "Diaphragm Wall",
                    "Revetment",
                    "Revetment",
                ],
                ("class", ""): [
                    "combinable",
                    "combinable",
                    "combinable",
                    "combinable",
                    "combinable",
                    "combinable",
                    "combinable",
                    "combinable",
                    "full",
                    "full",
                    "full",
                    "full",
                    "partial",
                    "full",
                    "revetment",
                    "revetment",
                ],
                ("year", ""): [0, 20, 0, 20, 0, 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0],
                ("dcrest", ""): [
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    -999,
                    -999,
                    -999,
                    -999,
                ],
                ("dberm", ""): [
                    0,
                    0,
                    5,
                    5,
                    0,
                    0,
                    5,
                    5,
                    0,
                    5,
                    0,
                    5,
                    -999,
                    -999,
                    -999,
                    -999,
                ],
                ("beta_target", ""): [
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    4.648,
                    4.648,
                ],
                ("transition_level", ""): [
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    -999,
                    3.84,
                    4.84,
                ],
                ("cost", ""): [
                    193369.0,
                    193369.0,
                    753520.5,
                    753520.5,
                    3324623.6,
                    3324623.6,
                    3884775.0,
                    3884775.0,
                    4950229.0,
                    5510380.0,
                    8081483.0,
                    8641635.0,
                    1302200.0,
                    26189540.0,
                    100000.0,
                    200000.0,
                ],
                ("OVERFLOW", 0): [
                    4.331,
                    4.331,
                    4.331,
                    4.331,
                    5.319,
                    4.331,
                    5.319,
                    4.331,
                    4.331,
                    4.331,
                    5.319,
                    5.319,
                    4.331,
                    5.374,
                    4.331,
                    4.331,
                ],
                ("OVERFLOW", 50): [
                    3.525,
                    3.525,
                    3.525,
                    3.525,
                    4.822,
                    4.822,
                    4.822,
                    4.822,
                    3.525,
                    3.525,
                    4.822,
                    4.822,
                    3.525,
                    4.919,
                    3.525,
                    3.525,
                ],
                ("STABILITY_INNER", 0): [
                    5.552,
                    5.552,
                    6.202,
                    5.552,
                    5.552,
                    5.552,
                    6.202,
                    5.552,
                    6.810,
                    7.460,
                    6.810,
                    7.460,
                    5.552,
                    7.034,
                    5.552,
                    5.552,
                ],
                ("STABILITY_INNER", 50): [
                    5.552,
                    5.552,
                    6.202,
                    6.202,
                    5.552,
                    5.552,
                    6.202,
                    5.552,
                    6.810,
                    7.460,
                    6.810,
                    7.460,
                    5.552,
                    7.034,
                    5.552,
                    5.552,
                ],
                ("PIPING", 0): [
                    3.185,
                    3.185,
                    3.282,
                    3.185,
                    3.185,
                    3.185,
                    3.282,
                    3.185,
                    3.185,
                    3.282,
                    3.185,
                    3.282,
                    4.818,
                    6.734,
                    3.185,
                    3.185,
                ],
                ("PIPING", 50): [
                    2.885,
                    2.885,
                    2.983,
                    2.983,
                    2.885,
                    2.885,
                    2.983,
                    2.983,
                    2.885,
                    2.983,
                    2.885,
                    2.983,
                    4.616,
                    6.6,
                    2.885,
                    2.885,
                ],
                ("REVETMENT", 0): [
                    4.6,
                    4.6,
                    4.6,
                    4.6,
                    4.6,
                    4.6,
                    4.6,
                    4.6,
                    4.6,
                    4.6,
                    4.6,
                    4.6,
                    4.6,
                    4.6,
                    4.6,
                    4.638,
                ],
                ("REVETMENT", 50): [
                    4.598,
                    4.598,
                    4.598,
                    4.598,
                    4.598,
                    4.598,
                    4.598,
                    4.598,
                    4.598,
                    4.598,
                    4.598,
                    4.598,
                    4.598,
                    4.598,
                    4.598,
                    4.638,
                ],
            }
        )

    def _create_solutions(self, section_name: str) -> Solutions:
        _config = VrtoolConfig()
        _solutions = Solutions(
            config=_config, dike_section=self._create_valid_dike_section(section_name)
        )
        _solutions.MeasureData = self._create_measures_df()
        return _solutions

    def _create_solution_dict(self):
        _solutions1 = self._create_solutions(SECTION1)
        _solutions2 = self._create_solutions(SECTION2)
        return {SECTION1: _solutions1, SECTION2: _solutions2}

    def test_mapping_input(self):
        # 1. Define input
        _selected_dike_traject = self._create_valid_dike_traject()
        _solutions_dict = self._create_solution_dict()

        # 2. Run test
        _optimization_controller = StrategyController("Dummy", VrtoolConfig())
        _optimization_controller.map_input(_selected_dike_traject, _solutions_dict)

        # 3. Verify expectations
        _sections = _optimization_controller._section_measures_input
        assert len(_sections) == 2
        assert _sections[0].section_name == SECTION1
        assert _sections[0].traject_name == TRAJECT1
        # Sh measures
        assert len(_sections[0].sh_measures) == 10

        assert isinstance(_sections[0].sh_measures[0], ShMeasure)
        assert (
            len(_sections[0].sh_measures[0].mechanism_year_collection.probabilities)
            == 4
        )
        # Sg measures
        assert len(_sections[0].sg_measures) == 10
        assert isinstance(_sections[0].sg_measures[0], SgMeasure)
        assert (
            len(_sections[0].sh_measures[0].mechanism_year_collection.probabilities)
            == 4
        )

    def test_mapping_output(self):
        # 1. Define input

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
            measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
            combine_type=CombinableTypeEnum.COMBINABLE,
            cost=193369,
            discount_rate=0.03,
            year=0,
            mechanism_year_collection=_mech_yr_coll_sh_soil_0,
            dcrest=0,
            beta_target=-999,
            transition_level=-999,
            start_cost=0,
        )
        _sh_measure_soil_20_0 = ShMeasure(
            measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
            combine_type=CombinableTypeEnum.COMBINABLE,
            cost=193369,
            discount_rate=0.03,
            year=20,
            mechanism_year_collection=_mech_yr_coll_sh_soil_0,
            dcrest=0,
            beta_target=-999,
            transition_level=-999,
            start_cost=0,
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
            measure_type=MeasureTypeEnum.DIAPHRAGM_WALL,
            combine_type=CombinableTypeEnum.FULL,
            cost=234567,
            discount_rate=0.03,
            year=0,
            mechanism_year_collection=_mech_yr_coll_sh_diaphragm_0,
            dcrest=-999,
            beta_target=-999,
            transition_level=-999,
            start_cost=1357903,
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
            measure_type=MeasureTypeEnum.REVETMENT,
            combine_type=CombinableTypeEnum.REVETMENT,
            cost=123456,
            discount_rate=0.03,
            year=0,
            mechanism_year_collection=_mech_yr_coll_sh_revetment_0_384,
            dcrest=-999,
            beta_target=4.648,
            transition_level=3.84,
            start_cost=0,
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
            measure_type=MeasureTypeEnum.REVETMENT,
            combine_type=CombinableTypeEnum.REVETMENT,
            cost=223456,
            discount_rate=0.03,
            year=0,
            mechanism_year_collection=_mech_yr_coll_sh_revetment_0_484,
            dcrest=-999,
            beta_target=4.648,
            transition_level=4.84,
            start_cost=0,
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
            measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
            combine_type=CombinableTypeEnum.COMBINABLE,
            cost=193369,
            discount_rate=0.03,
            year=0,
            mechanism_year_collection=_mech_yr_coll_sg_soil_0,
            dberm=0,
            dcrest=0,
            start_cost=193369,
        )
        _sg_measure_soil_20_0 = SgMeasure(
            measure_type=MeasureTypeEnum.SOIL_REINFORCEMENT,
            combine_type=CombinableTypeEnum.COMBINABLE,
            cost=193369,
            discount_rate=0.03,
            year=20,
            mechanism_year_collection=_mech_yr_coll_sg_soil_0,
            dberm=0,
            dcrest=0,
            start_cost=193369,
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
            measure_type=MeasureTypeEnum.VERTICAL_GEOTEXTILE,
            combine_type=CombinableTypeEnum.PARTIAL,
            cost=1302200,
            discount_rate=0.03,
            year=0,
            mechanism_year_collection=_mech_yr_coll_sg_vzg_0,
            dberm=-999,
            dcrest=-999,
            start_cost=0,
        )

        # Controller
        _optimization_controller = StrategyController("Dummy", VrtoolConfig())
        _optimization_controller._section_measures_input = [
            SectionAsInput(
                section_name=SECTION1,
                traject_name=TRAJECT1,
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
            )
        ]

        # Combinations
        _sh_combination_soil_0 = CombinedMeasure.from_input(_sh_measure_soil_0_0, None)
        _sh_combination_soil_20 = CombinedMeasure.from_input(
            _sh_measure_soil_20_0, None
        )
        _sh_combination_soil_revetment_0_384 = CombinedMeasure.from_input(
            _sh_measure_soil_0_0, _sh_measure_revetment_0_384
        )
        _sh_combination_soil_revetment_20_384 = CombinedMeasure.from_input(
            _sh_measure_soil_20_0, _sh_measure_revetment_0_384
        )
        _sh_combination_soil_revetment_0_484 = CombinedMeasure.from_input(
            _sh_measure_soil_0_0, _sh_measure_revetment_0_484
        )
        _sh_combination_soil_revetment_20_484 = CombinedMeasure.from_input(
            _sh_measure_soil_20_0, _sh_measure_revetment_0_484
        )
        _sh_combination_diaphragm_0 = CombinedMeasure.from_input(
            _sh_measure_diaphragm_0, None
        )
        _sh_comb = 5
        _sg_combination_soil_0 = CombinedMeasure.from_input(_sg_measure_soil_0_0, None)
        _sg_combination_soil_20 = CombinedMeasure.from_input(
            _sg_measure_soil_20_0, None
        )
        _sg_combination_soil_vzg_0 = CombinedMeasure.from_input(
            _sg_measure_soil_0_0, _sg_measure_vzg_0
        )
        _sg_combination_soil_vzg_20 = CombinedMeasure.from_input(
            _sg_measure_soil_20_0, _sg_measure_vzg_0
        )
        _sg_comb = 4
        _optimization_controller._section_measures_input[0].combined_measures = [
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
        _optimization_controller._section_measures_input[
            0
        ].aggregated_measure_combinations = [
            AggregatedMeasureCombination(
                _sh_combination_soil_0, _sg_combination_soil_0, 0
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_0_384, _sg_combination_soil_0, 0
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_0_484, _sg_combination_soil_0, 0
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_0, _sg_combination_soil_vzg_0, 0
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_0_384, _sg_combination_soil_vzg_0, 0
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_0_484, _sg_combination_soil_vzg_0, 0
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_20, _sg_combination_soil_20, 20
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_20_384, _sg_combination_soil_20, 20
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_20_484, _sg_combination_soil_20, 20
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_20, _sg_combination_soil_vzg_20, 20
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_20_384, _sg_combination_soil_vzg_20, 20
            ),
            AggregatedMeasureCombination(
                _sh_combination_soil_revetment_20_484, _sg_combination_soil_vzg_20, 20
            ),
        ]

        # 2. Run test
        _optimization_controller.map_output()

        # 3. Verify expectations

        # Probabilities
        assert _optimization_controller.Pf is not None
        assert _optimization_controller.Pf[MechanismEnum.OVERFLOW.name].shape == (
            1,
            8,
            50,
        )
        assert _optimization_controller.Pf[MechanismEnum.REVETMENT.name].shape == (
            1,
            8,
            50,
        )
        assert _optimization_controller.Pf[MechanismEnum.PIPING.name].shape == (
            1,
            5,
            50,
        )
        assert _optimization_controller.Pf[
            MechanismEnum.STABILITY_INNER.name
        ].shape == (
            1,
            5,
            50,
        )
        assert _optimization_controller.Pf[MechanismEnum.STABILITY_INNER.name][
            0, 0, 0
        ] == pytest.approx(0.0010)

        # Cost
        assert _optimization_controller.LCCOptions is not None
        assert _optimization_controller.LCCOptions.shape == (1, 8, 5)
        assert _optimization_controller.LCCOptions[0, 0, 0] == pytest.approx(0.0)
        assert _optimization_controller.LCCOptions[0, 1, 1] == pytest.approx(193369.0)
        assert _optimization_controller.LCCOptions[0, 1, 2] == pytest.approx(1e99)
        assert _optimization_controller.LCCOptions[0, 1, 3] == pytest.approx(1495569.0)
        assert _optimization_controller.LCCOptions[0, 2, 2] == pytest.approx(
            107063.7269
        )
        assert _optimization_controller.LCCOptions[0, 3, 3] == pytest.approx(1619025.0)
        assert _optimization_controller.LCCOptions[0, 4, 4] == pytest.approx(
            1532719.7269
        )

        # Other structures
        assert _optimization_controller.Cint_h.shape == (1, 7)
        assert _optimization_controller.Cint_g.shape == (1, 4)
        assert _optimization_controller.Dint.shape == (1, 7)
        assert _optimization_controller.D.shape == (50,)
        assert _optimization_controller.RiskGeotechnical.shape == (1, 5, 50)
        assert _optimization_controller.RiskOverflow.shape == (1, 8, 50)
        assert _optimization_controller.RiskRevetment.shape == (1, 8, 50)
