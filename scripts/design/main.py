from pathlib import Path

import numpy as np

from scripts.design.modify_inputs import copy_database, modify_beta_measure_database
from scripts.design.run_vrtool_specific import run_dsn_lenient_and_stringent, rerun_database
from vrtool.api import ApiRunWorkflows
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.models import MeasureResultSection, MeasureResult, MeasurePerSection, Measure, \
    ComputationScenarioParameter
from vrtool.orm.orm_controllers import get_all_measure_results_with_supported_investment_years, open_database
from time import time
import shutil

_input_model = Path(
    r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\38-1\test")
assert _input_model.exists()
_vr_config = VrtoolConfig().from_json(Path(_input_model).joinpath("config.json"))
_vr_config.input_directory = _input_model

copy_database(_vr_config, "modified_beta_Section")
modify_beta_measure_database(_vr_config, measure_type="soil reinforcement", mechanism="Overflow", std_beta=1.0)
# modify_cost_measure_database(_vr_config, multiplier=2, measure_type="soil reinforcement")
# modify_initial_beta_stability(_vr_config, std_beta=1.0)
# rerun_database(_vr_config, rerun_all=True)
# run_dsn_lenient_and_stringent(_vr_config, run_strict=False)


