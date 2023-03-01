from pathlib import Path

import pandas as pd
import pytest

from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem.DikeTraject import DikeTraject
from src.run_workflows.measures_workflow.results_measures import ResultsMeasures
from src.run_workflows.measures_workflow.run_measures import RunMeasures
from src.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)
from src.run_workflows.optimization_workflow.run_optimization import RunOptimization
from src.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)
from src.run_workflows.safety_workflow.run_safety_assessment import RunSafetyAssessment
from src.run_workflows.vrtool_plot_mode import VrToolPlotMode
from src.run_workflows.vrtool_run_model import run_model_old_approach
from tests import get_test_results_dir, test_data
"""This is a test based on 10 sections from traject 16-4 of the SAFE project"""


class TestAcceptance:
    @pytest.mark.parametrize(
        "casename, traject",
        [
            ("integrated_SAFE_16-3_small", "16-3"),
            ("TestCase1_38-1_no_housing", "38-1"),
            ("TestCase2_38-1_overflow_no_housing", "38-1"),
        ],
    )
    def test_run_as_sandbox(
        self, casename: str, traject: str, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _test_dir = test_data / casename
        assert _test_dir.exists(), "No input data found at {}".format(_test_dir)

        _results_dir = get_test_results_dir(request) / casename
        _vr_config = VrtoolConfig()
        _vr_config.input_directory = _test_dir
        _vr_config.output_directory = _results_dir
        _vr_config.traject = traject
        _plot_mode = VrToolPlotMode.STANDARD

        # Initialize output sub folders
        (_vr_config.output_directory / "figures").mkdir(parents=True)
        (_vr_config.output_directory / "results" / "investments_steps").mkdir(
            parents=True
        )

        # 2. Run test.
        # Step 0. Load Traject
        _selected_traject = DikeTraject.from_vr_config(_vr_config)
        assert isinstance(_selected_traject, DikeTraject)

        # Step 1. Safety assessment.
        _safety_assessment = RunSafetyAssessment(_vr_config, _selected_traject, plot_mode=_plot_mode)
        _safety_result = _safety_assessment.run()
        assert isinstance(_safety_result, ResultsSafetyAssessment)

        # Step 2. Measures.
        _measures = RunMeasures(_vr_config, _selected_traject, plot_mode=_plot_mode)
        _measures_result = _measures.run()
        assert isinstance(_measures_result, ResultsMeasures)

        # Step 3. Optimization.
        _optimization = RunOptimization(_measures_result, plot_mode=_plot_mode)
        _optimization_result = _optimization.run()
        assert isinstance(_optimization_result, ResultsOptimization)

        # 3. Verify expectations.
        _found_errors = []
        files_to_compare = [
            "TakenMeasures_Doorsnede-eisen.csv",
            "TakenMeasures_Veiligheidsrendement.csv",
            "TotalCostValues_Greedy.csv",
        ]

        for _f_to_compare in files_to_compare:
            reference = pd.read_csv(
                _test_dir / "reference" / "results" / _f_to_compare, index_col=0
            )
            result = pd.read_csv(_results_dir / "results" / _f_to_compare, index_col=0)
            if not reference.equals(result):
                _found_errors.append("{} is different.".format(_f_to_compare))

        # assert no error message has been registered, else print messages
        assert not _found_errors, "errors occured:\n{}".format("\n".join(_found_errors))

    @pytest.mark.parametrize(
        "casename, traject",
        [
            ("integrated_SAFE_16-3_small", "16-3"),
            ("TestCase1_38-1_no_housing", "38-1"),
            ("TestCase2_38-1_overflow_no_housing", "38-1"),
        ],
    )
    def test_integrated_run(self, casename, traject, request: pytest.FixtureRequest):
        """This test so far only checks the output values after optimization.
        The test should eventually e split for the different steps in the computation (assessment, measures and optimization)"""
        test_data_input_directory = Path.joinpath(test_data, casename)
        test_results_dir = get_test_results_dir(request).joinpath(casename)

        _test_config = VrtoolConfig()
        _test_config.input_directory = test_data_input_directory
        _test_config.output_directory = test_results_dir

        _test_config.output_directory.joinpath("figures").mkdir(parents=True)
        _test_config.output_directory.joinpath("results", "investment_steps").mkdir(
            parents=True
        )

        _test_traject = DikeTraject(_test_config, traject=traject)
        _test_traject.ReadAllTrajectInput(input_path=test_data_input_directory)

        AllStrategies, AllSolutions = run_model_old_approach(_test_config, _test_traject)

        comparison_errors = []
        files_to_compare = [
            "TakenMeasures_Doorsnede-eisen.csv",
            "TakenMeasures_Veiligheidsrendement.csv",
            "TotalCostValues_Greedy.csv",
        ]

        reference_path = Path.joinpath(test_data, casename, "reference")
        for file in files_to_compare:
            reference = pd.read_csv(
                reference_path.joinpath("results", file), index_col=0
            )
            result = pd.read_csv(
                test_results_dir.joinpath("results", file), index_col=0
            )
            if not reference.equals(result):
                comparison_errors.append("{} is different.".format(file))

        # assert no error message has been registered, else print messages
        assert not comparison_errors, "errors occured:\n{}".format(
            "\n".join(comparison_errors)
        )
