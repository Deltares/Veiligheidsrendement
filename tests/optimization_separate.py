import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from tests import get_test_results_dir, test_data, test_externals
from tests.test_acceptance import TestAcceptance
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject, calc_traject_prob
from vrtool.orm.orm_controllers import get_dike_traject
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.optimization_workflow.run_optimization import RunOptimization
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)
from vrtool.run_workflows.safety_workflow.run_safety_assessment import (
    RunSafetyAssessment,
)
from vrtool.run_workflows.vrtool_plot_mode import VrToolPlotMode
from vrtool.run_workflows.vrtool_run_full_model import RunFullModel

_casename = "TestCase1_38-1_no_housing"
_traject = "38-1"
_test_input_directory = Path.joinpath(test_data, _casename)
assert _test_input_directory.exists()

_test_results_directory = (
    Path(
        r"C:\Repositories\VRSuite\VRTool\master\tests\test_results\test_run_optimization"
    )
    / _casename
)
if _test_results_directory.exists():
    shutil.rmtree(_test_results_directory)

valid_vrtool_config = VrtoolConfig()
valid_vrtool_config.input_directory = _test_input_directory
valid_vrtool_config.output_directory = _test_results_directory
valid_vrtool_config.traject = _traject
valid_vrtool_config.externals = test_externals
valid_vrtool_config.input_database_path = _test_input_directory.joinpath(
    "vrtool_input.db"
)

_test_reference_path = valid_vrtool_config.input_directory / "reference"

_shelve_path = valid_vrtool_config.input_directory / "shelves"
_results_assessment = ResultsSafetyAssessment()
_results_assessment.load_results(alternative_path=_shelve_path / "AfterStep1.out")
_results_measures = ResultsMeasures()

_results_measures.vr_config = valid_vrtool_config
_results_measures.selected_traject = _results_assessment.selected_traject

_results_measures.load_results(alternative_path=_shelve_path / "AfterStep2.out")
_results_optimization = RunOptimization(_results_measures, valid_vrtool_config).run()

TestAcceptance._validate_acceptance_result_cases(
    valid_vrtool_config.output_directory, test_reference_dir=_test_reference_path
)
