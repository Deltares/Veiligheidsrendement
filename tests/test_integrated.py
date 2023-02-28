from pathlib import Path

import pandas as pd
import pytest

from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem.DikeTraject import DikeTraject
from src.run_workflows.measures_workflow.run_measures import RunMeasures
from src.run_workflows.optimization_workflow.run_optimization import RunOptimization
from src.run_workflows.safety_workflow.run_safety_assessment import RunSafetyAssessment
from src.run_workflows.vrtool_plot_mode import VrToolPlotMode
from tests import get_test_results_dir, test_data
from tools.RunModel import runFullModel

"""This is a test based on 10 sections from traject 16-4 of the SAFE project"""


class TestAcceptance:
    @pytest.mark.parametrize(
        "casename, traject", [("integrated_SAFE_16-3_small", "16-3")]
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
        _safety_assessment = RunSafetyAssessment(plot_mode=_plot_mode)
        _safety_assessment.selected_traject = _selected_traject
        _safety_assessment.vr_config = _vr_config
        _safety_result = _safety_assessment.run()

        # Step 2. Measures.
        _measures = RunMeasures(plot_mode=_plot_mode)
        _measures.selected_traject = _selected_traject
        _measures.vr_config = _vr_config
        _measures_result = _measures.run()

        # Step 3. Optimization.
        _optimization = RunOptimization(_measures_result, plot_mode=_plot_mode)
        _optimization_result = _optimization.run()

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
        "casename, traject", [("integrated_SAFE_16-3_small", "16-3")]
    )
    def test_integrated_run(self, casename, traject, request: pytest.FixtureRequest):
        """This test so far only checks the output values after optimization.
        The test should eventually e split for the different steps in the computation (assessment, measures and optimization)"""
        test_data_input_directory = Path.joinpath(test_data, casename)
        test_results_dir = get_test_results_dir(request).joinpath(casename)

        test_config = VrtoolConfig()
        test_config.input_directory = test_data_input_directory
        test_config.directory = test_results_dir

        test_config.directory.joinpath("figures").mkdir(parents=True)
        test_config.directory.joinpath("results", "investment_steps").mkdir(
            parents=True
        )

        TestTrajectObject = DikeTraject(test_config, traject=traject)
        TestTrajectObject.ReadAllTrajectInput(input_path=test_data_input_directory)

        AllStrategies, AllSolutions = runFullModel(TestTrajectObject, test_config)

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
