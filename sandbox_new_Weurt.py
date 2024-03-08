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
import cProfile
import re

# ====== IMPORTANT ======== #
# The initial stix must be pre-processed before using them with the prototype. They need to be executed blankly first
# otherwise the serialization will fail.

# 1. Define input and output directories..

_input_model = Path(
    r"c:\Users\klerk_wj\OneDrive - Stichting Deltares\00_Projecten\11_VR_Samenwerken aan Kunstwerken\Testcases\41-1_compleet"
)
_results_dir = Path(
    r"c:\Users\klerk_wj\OneDrive - Stichting Deltares\00_Projecten\11_VR_Samenwerken aan Kunstwerken\Testcases\41-1_compleet\basisrun"
)

# _input_model = Path(
#     r"c:\Users\klerk_wj\OneDrive - Stichting Deltares\00_Projecten\11_VR_Samenwerken aan Kunstwerken\Testcases\41-1_dummy\v0.1.3"
# )
# _results_dir = Path(
#     r"c:\Users\klerk_wj\OneDrive - Stichting Deltares\00_Projecten\11_VR_Samenwerken aan Kunstwerken\Testcases\41-1_dummy\v0.1.3\basisrun"
# )

# 2. Define the configuration to use.
_vr_config = VrtoolConfig()
_vr_config.input_directory = _input_model
# _vr_config.excluded_mechanisms = [MechanismEnum.HYDRAULIC_STRUCTURES]
_vr_config.excluded_mechanisms = [MechanismEnum.REVETMENT, MechanismEnum.HYDRAULIC_STRUCTURES]
_vr_config.output_directory = _results_dir
_vr_config.externals = (
    Path(__file__).parent.parent / "externals/D-Stability 2022.01.2/bin"
)
_vr_config.traject = "41-1"
_vr_config.design_methods = ["Veiligheidsrendement"]

_vr_config.input_database_name = "41-1_met_weurt.db"
# _vr_config.input_database_name = "41-1_database.db"

api = ApiRunWorkflows(vrtool_config=_vr_config)
VrToolLogger.init_console_handler(logging.INFO)

#do a full run


#evaluate measures
# results_measures = api.run_measures()

# #only optimization:
clear_optimization_results(_vr_config)

# Basisberekening
selected_measures = (
    get_all_measure_results_with_supported_investment_years(_vr_config)
)

#aanpassingen voor Sluis Weurt:
aantal_maatregelen = 8
weurt_investment_year = 20

work_dir = Path(r'c:\Users\klerk_wj\OneDrive - Stichting Deltares\00_Projecten\11_VR_Samenwerken aan Kunstwerken\Testcases')


#CASE 2: BASISVARIANT MET WEURT
selected_measures = (
    get_all_measure_results_with_supported_investment_years(_vr_config)
)
selected_measures1 = [mrid for mrid in selected_measures if mrid[0]< 3088]
structure_measures = [(3160, 0), (3161, 0), (3162, 0) ,(3163, 0), (3164, 0), (3165, 0), (3166, 0), (3167, 0)]
selected_measures2 = [mrid for mrid in selected_measures if mrid[0]> 3234]
selected_measures = selected_measures1 + structure_measures + selected_measures2

_vr_config.output_directory = _results_dir.parent.joinpath('Met Weurt')
results_optimization = api.run_optimization_with_structure('Met sluis Weurt', selected_measures, 'Sluis Weurt', 
                                                           work_dir.joinpath('Weurt_maatregelen.xlsx'))
#CASE 1: BASISVARIANT GEEN INVLOED WEURT
#MeasureResultIDs are 3088-3234
selected_measures1 = [mrid for mrid in selected_measures if mrid[0]< 3088]
structure_measures = [(3160, 0)] * 1
selected_measures2 = [mrid for mrid in selected_measures if mrid[0]> 3234]
selected_measures = selected_measures1 + structure_measures + selected_measures2

_vr_config.output_directory = _results_dir.parent.joinpath('Zonder Weurt')
results_optimization = api.run_optimization_with_structure('Zonder sluis Weurt', selected_measures, 'Sluis Weurt', 
                                                           work_dir.joinpath('Weurt_maatregelen_geen_effect.xlsx'))

#CASE 3: VARIANT MET WEURT LAAT
selected_measures = (
    get_all_measure_results_with_supported_investment_years(_vr_config)
)
selected_measures1 = [mrid for mrid in selected_measures if mrid[0]< 3088]
structure_measures = [(3160, weurt_investment_year), (3161, weurt_investment_year), (3162, weurt_investment_year) ,
                      (3163, weurt_investment_year), (3164, weurt_investment_year), (3165, weurt_investment_year) , 
                      (3166, weurt_investment_year), (3167, weurt_investment_year)]

selected_measures2 = [mrid for mrid in selected_measures if mrid[0]> 3234]
selected_measures = selected_measures1 + structure_measures + selected_measures2

_vr_config.output_directory = _results_dir.parent.joinpath('Met Weurt laat')
results_optimization = api.run_optimization_with_structure('Sluis Weurt in 2045', selected_measures, 'Sluis Weurt', 
                                                           work_dir.joinpath('Weurt_maatregelen.xlsx'))


# 
# cProfile.run('api.run_optimization("Basisberekening_tijdtest", selected_measures)', 'optimization_profiling.prof')


# #dummy case:
# #MeasureResultID 148-294 are for Weurt
# selected_measures = [mrid for mrid in selected_measures if mrid[0]< 148]
# #structure measures (to ensure size is appropriate):
# structure_measures = [(148, 10)] * 10
# selected_measures = selected_measures + structure_measures
