import os
import sys
from pathlib import Path

import pandas as pd
import pytest

'''This is a test based on 10 sections from traject 16-4 of the SAFE project'''

@pytest.mark.parametrize("casename",['integrated_SAFE_16-3_small'])
def test_integrated_run(casename):
    '''This test so far only checks the output values after optimization.
    The test should eventually e split for the different steps in the computation (assessment, measures and optimization)'''
    sys.path.append(os.getcwd() + '\\{}'.format(casename))
    import config_test as case_config

    sys.path.append('../../src')
    from FloodDefenceSystem.DikeTraject import DikeTraject
    from tools.RunModel import runFullModel

    TestTrajectObject = DikeTraject(traject=case_config.traject)

    TestTrajectObject.ReadAllTrajectInput(input_path=case_config.path)

    AllStrategies, AllSolutions = runFullModel(TestTrajectObject,case_config)

    reference_path = Path(os.getcwd() + '\\{}\\reference'.format(casename))

    comparison_errors = []

    files_to_compare = ['TakenMeasures_Doorsnede-eisen.csv',
                        'TakenMeasures_Veiligheidsrendement.csv',
                        'TotalCostValues_Greedy.csv']
    for file in files_to_compare:
        reference = pd.read_csv(reference_path.joinpath('results',file),index_col=0)
        result    = pd.read_csv(case_config.directory.joinpath('results',file),index_col=0)
        if not reference.equals(result):
            comparison_errors.append('{} is different.'.format(file))

    # assert no error message has been registered, else print messages
    assert not comparison_errors, "errors occured:\n{}".format("\n".join(comparison_errors))
