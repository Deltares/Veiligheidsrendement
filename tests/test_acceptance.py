from pathlib import Path
import shutil

import pandas as pd
import pytest

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.FloodDefenceSystem.DikeTraject import DikeTraject
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)
from vrtool.run_workflows.optimization_workflow.run_optimization import RunOptimization
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)
from vrtool.run_workflows.safety_workflow.run_safety_assessment import RunSafetyAssessment
from vrtool.run_workflows.vrtool_plot_mode import VrToolPlotMode
from vrtool.run_workflows.vrtool_run_full_model import RunFullModel
from tests import get_test_results_dir, test_data

"""This is a test based on 10 sections from traject 16-4 of the SAFE project"""
_acceptance_test_cases = [
    pytest.param("integrated_SAFE_16-3_small", "16-3"),
    pytest.param("TestCase1_38-1_no_housing", "38-1"),
    pytest.param("TestCase2_38-1_overflow_no_housing", "38-1"),
]


@pytest.mark.slow
class TestAcceptance:
    def _validate_acceptance_result_cases(self, test_results_dir: Path, test_reference_dir: Path):
        comparison_errors = []
        files_to_compare = [
            "TakenMeasures_Doorsnede-eisen.csv",
            "TakenMeasures_Veiligheidsrendement.csv",
            "TotalCostValues_Greedy.csv",
        ]

        for file in files_to_compare:
            reference = pd.read_csv(
                test_reference_dir.joinpath("results", file), index_col=0
            )
            result = pd.read_csv(
                test_results_dir / file, index_col=0
            )
            if not reference.equals(result):
                comparison_errors.append("{} is different.".format(file))

        # assert no error message has been registered, else print messages
        assert not comparison_errors, "errors occured:\n{}".format(
            "\n".join(comparison_errors)
        )

    @pytest.mark.parametrize(
        "casename, traject", _acceptance_test_cases,
    )
    def test_run_as_sandbox(
        self, casename: str, traject: str, request: pytest.FixtureRequest
    ):
        """
        This test so far only checks the output values after optimization.
        """
        # 1. Define test data.
        _test_dir = test_data / casename
        assert _test_dir.exists(), "No input data found at {}".format(_test_dir)

        _results_dir = get_test_results_dir(request) / casename
        if _results_dir.exists():
            shutil.rmtree(_results_dir)
        _vr_config = VrtoolConfig()
        _vr_config.input_directory = _test_dir
        _vr_config.output_directory = _results_dir
        _vr_config.traject = traject
        _plot_mode = VrToolPlotMode.STANDARD

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
        self._validate_acceptance_result_cases(_results_dir, _test_dir / "reference")


    @pytest.mark.parametrize(
        "casename, traject",_acceptance_test_cases,
    )
    def test_run_full_model(self, casename: str, traject: str, request: pytest.FixtureRequest):
        """
        This test so far only checks the output values after optimization.
        """
        # 1. Define test data.
        _test_input_directory = Path.joinpath(test_data, casename)
        assert _test_input_directory.exists()

        _test_results_directory = get_test_results_dir(request).joinpath(casename)
        if _test_results_directory.exists():
            shutil.rmtree(_test_results_directory)

        _test_reference_path = _test_input_directory / "reference"
        assert _test_reference_path.exists()

        _test_config = VrtoolConfig()
        _test_config.input_directory = _test_input_directory
        _test_config.output_directory = _test_results_directory
        _test_config.traject = traject
        _test_traject = DikeTraject.from_vr_config(_test_config)

        # 2. Run test.
        RunFullModel(_test_config, _test_traject, VrToolPlotMode.STANDARD).run()

        # 3. Verify final expectations.
        self._validate_acceptance_result_cases(_test_results_directory, _test_reference_path)
