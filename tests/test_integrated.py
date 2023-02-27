from pathlib import Path

import pandas as pd
import pytest

from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem.DikeTraject import DikeTraject
from tests import get_test_results_dir, test_data
from tools.RunModel import runFullModel

"""This is a test based on 10 sections from traject 16-4 of the SAFE project"""


class TestAcceptance:
    @pytest.mark.parametrize(
        "casename, traject", [("integrated_SAFE_16-3_small", "16-3")]
    )
    def test_integrated_run(self, casename, traject, request: pytest.FixtureRequest):
        """This test so far only checks the output values after optimization.
        The test should eventually e split for the different steps in the computation (assessment, measures and optimization)"""
        test_data_input_directory = Path.joinpath(test_data, casename)
        test_results_dir = get_test_results_dir(request)

        test_config = VrtoolConfig()
        test_config.input_directory = test_data_input_directory
        test_config.directory = test_results_dir

        # Make a few dirs if they dont exist yet:
        test_config.directory.joinpath("figures").mkdir(parents=True)
        test_config.directory.joinpath("results", "investment_steps").mkdir(parents=True)

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
