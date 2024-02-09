import pytest

from vrtool.optimization.measures.combined_measure import CombinedMeasure
from tests.optimization.controllers.test_strategy_controller import TestStrategyController
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.controllers.strategy_controller import StrategyController

class TestCombinedMeasure:
    def testCombine(self):
        # 1. Define input
        _strategyController = TestStrategyController();
        _selected_dike_traject = _strategyController._create_valid_dike_traject()
        _solutions_dict = _strategyController._create_solution_dict()
        _optimization_controller = StrategyController("Dummy", VrtoolConfig())
        _optimization_controller.map_input(_selected_dike_traject, _solutions_dict)
        pass
        _section_measures = _optimization_controller._section_measures_input
        _ms1 = _section_measures[0].sh_measures[0]
        _ms2 = _section_measures[1].sh_measures[0]
        _combined = CombinedMeasure(_ms1, _ms2)
        pass
