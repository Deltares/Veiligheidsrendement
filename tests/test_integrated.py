import os
import sys
from pathlib import Path

import pandas as pd
import pytest

import src.defaults.vrtool_config as case_config
import src.FloodDefenceSystem.DikeTraject as DikeTraject
import tools.RunModel as runFullModel
from src.defaults.vrtool_config import VrtoolConfig
from tests import test_data

"""This is a test based on 10 sections from traject 16-4 of the SAFE project"""


class TestAcceptance:
    @pytest.mark.parametrize(
        "casename", "traject", ["integrated_SAFE_16-3_small", "16-3"]
    )
    def test_integrated_run(self, casename, traject):
        """This test so far only checks the output values after optimization.
        The test should eventually e split for the different steps in the computation (assessment, measures and optimization)"""
        sys.path.append(os.getcwd() + "\\{}".format(casename))

        TestTrajectObject = DikeTraject(traject=traject)

        test_data_input_directory = Path.joinpath(test_data, casename)
        TestTrajectObject.ReadAllTrajectInput(input_path=test_data_input_directory)

        test_config = VrtoolConfig()
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
                case_config.directory.joinpath("results", file), index_col=0
            )
            if not reference.equals(result):
                comparison_errors.append("{} is different.".format(file))

        # assert no error message has been registered, else print messages
        assert not comparison_errors, "errors occured:\n{}".format(
            "\n".join(comparison_errors)
        )
