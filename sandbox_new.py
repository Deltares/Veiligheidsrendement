from pathlib import Path
from shutil import rmtree
from typing import Dict, List

from geolib import DStabilityModel
from shapely import Polygon
import random
from vrtool.api import run_step_optimization, ApiRunWorkflows
from vrtool.common.enums import MechanismEnum
from vrtool.decision_making.measures import SoilReinforcementMeasure
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.orm.models import MeasureResult
from vrtool.orm.orm_controllers import export_results_safety_assessment, get_dike_traject, clear_assessment_results, \
    clear_measure_results, export_results_measures, clear_optimization_results, get_exported_measure_result_ids, \
    export_results_optimization, get_all_measure_results_with_supported_investment_years
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.safety_workflow.run_safety_assessment import (
    RunSafetyAssessment,
)
import logging
from vrtool.vrtool_logger import VrToolLogger
import pandas as pd

# ====== IMPORTANT ======== #
# The initial stix must be pre-processed before using them with the prototype. They need to be executed blankly first
# otherwise the serialization will fail.

# 1. Define input and output directories..

_input_model = Path(
    r"c:\Users\klerk_wj\OneDrive - Stichting Deltares\00_Projecten\11_VR_HWBP\00_testwerk\SITO_testcase"
)
_results_dir = Path(
    r"c:\Users\klerk_wj\OneDrive - Stichting Deltares\00_Projecten\11_VR_HWBP\00_testwerk\SITO_testcase\test1"
)


# 2. Define the configuration to use.
_vr_config = VrtoolConfig()
_vr_config.input_directory = _input_model
# _vr_config.excluded_mechanisms = [MechanismEnum.HYDRAULIC_STRUCTURES]
_vr_config.excluded_mechanisms = [MechanismEnum.REVETMENT, MechanismEnum.HYDRAULIC_STRUCTURES]
_vr_config.output_directory = _results_dir
_vr_config.externals = (
    Path(__file__).parent.parent / "externals/D-Stability 2022.01.2/bin"
)
_vr_config.traject = "38-1"

_vr_config.input_database_name = "38-1_test_SITO.db"


# if _results_dir.exists():
#     rmtree(_results_dir)

api = ApiRunWorkflows(vrtool_config=_vr_config)
VrToolLogger.init_console_handler(logging.INFO)

# BASISBEREKENING - ALLE STAPPEN
# results_optimization = api.run_all()

# #only optimization:
# # clear_optimization_results(_vr_config)

# BASISBEREKENING - ALLEEN OPTIMALISATIE
_vr_config.output_directory = _results_dir.parent.joinpath('Basisberekening_optimalisatie')
selected_measures = (
    get_all_measure_results_with_supported_investment_years(_vr_config)
)
results_optimization = api.run_optimization('Basisberekening - optimalisatie 2', selected_measures)

# OPTIMALISATIE - OP BASIS VAN CSV TODO: netjes maken
# #optimization with selection of measures:
# measure_data = pd.read_csv(_input_model.joinpath('year_settings.csv'))[['MeasureResultBegin','MeasureResultEind', 'jaar_variant2_met_werkendam']].dropna()
# measure_data['MeasureResultBegin'] = measure_data['MeasureResultBegin'].astype(int)
# measure_data['MeasureResultEind'] = measure_data['MeasureResultEind'].astype(int)
# measure_data['jaar_variant2_met_werkendam'] = measure_data['jaar_variant2_met_werkendam'].astype(int)
# selected_measures = []
# for count, section in measure_data.iterrows():
#     sections = list(range(section.MeasureResultBegin, section.MeasureResultEind+1))
#     years = [section.jaar_variant2_met_werkendam] * len(sections)
#     selected_measures = selected_measures + list(zip(sections,years))

# results_optimization = api.run_optimization('Variant 2 - met Werkendam', selected_measures)



# selected_measures = [(i, 0) for i in range(1, 100)]
# selected_measures = [(i, 14) for i in range(1, 163)] + [(i, 0) for i in range(163, 1000)]
# selected_measures = [(i, 0) for i in range(1, 1630)]

#timing of 1-163 to 
#profile this call using cProfile
# cProfile.run('results_optimization = api.run_optimization(selected_measures)', 'restats')
