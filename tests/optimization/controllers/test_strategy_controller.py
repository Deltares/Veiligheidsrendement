from pandas import DataFrame as df

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.optimization.controllers.strategy_controller import StrategyController
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

    def _create_measures(self) -> df:
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
                ],
                ("year", ""): [0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0],
                ("dcrest", ""): [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, -999],
                ("dberm", ""): [0, 0, 5, 5, 0, 0, 5, 5, 0, 5, 0, 5, -999],
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
                ],
            }
        )

    def _create_solutions(self, section_name: str) -> Solutions:
        _config = VrtoolConfig()
        _solutions = Solutions(
            config=_config, dike_section=self._create_valid_dike_section(section_name)
        )
        _solutions.MeasureData = self._create_measures()
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
        assert len(_sections[0].sh_measures) == 7

        assert isinstance(_sections[0].measures[0], ShMeasure)
        assert (
            len(_sections[0].measures[0].mechanism_year_collection._probabilities) == 2
        )
        # Sg measures
        assert len(_sections[0].sg_measures) == 7
        assert isinstance(_sections[0].measures[1], SgMeasure)
        assert (
            len(_sections[0].measures[1].mechanism_year_collection._probabilities) == 4
        )

    def test_combining_measures(self):
        # 1. Define input
        pass

        # 2. Run test

        # 3. Verify expectations
