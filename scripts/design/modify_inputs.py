from pathlib import Path

import numpy as np

from scripts.design.run_vrtool_specific import run_dsn_lenient_and_stringent, rerun_database
from vrtool.api import ApiRunWorkflows
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.models import MeasureResultSection, MeasureResult, MeasurePerSection, Measure, \
    ComputationScenarioParameter
from vrtool.orm.orm_controllers import get_all_measure_results_with_supported_investment_years, open_database
from time import time
import shutil


def modify_initial_beta_stability(_vr_config: VrtoolConfig, std_beta: float):
    """
    Modify the betas for stability only in the initial assessment.
    Apply a log-normal distribution of the betas

    Args:
        _vr_config:
        std_beta: standard deviation of a normal distribution to vary the betas.

    Returns:

    """
    _connected_db = open_database(_vr_config.input_database_path)

    query = ComputationScenarioParameter.select().where(ComputationScenarioParameter.parameter == "beta")
    for row in query:
        new_beta = np.random.normal(row.value, std_beta)
        row.value = new_beta
        row.save()
    return


def modify_cost_measure_database(_vr_config: VrtoolConfig, multiplier: int = 2, measure_type: str = None):
    # Open the database and reduce cost of the measures:
    _connected_db = open_database(_vr_config.input_database_path)

    if measure_type == "soil reinforcement":
        subquery = (MeasureResult
                    .select(MeasureResult.id)
                    .join(MeasurePerSection)
                    .join(Measure)
                    .where(Measure.measure_type_id.in_([1, 2])))
    elif measure_type == "vertical piping solution":
        subquery = (MeasureResult
                    .select(MeasureResult.id)
                    .join(MeasurePerSection)
                    .join(Measure)
                    .where(Measure.measure_type_id == 3))
    elif measure_type == "diaphram wall":
        subquery = (MeasureResult
                    .select(MeasureResult.id)
                    .join(MeasurePerSection)
                    .join(Measure)
                    .where(Measure.measure_type_id == 4))
    elif measure_type == "stability screening":
        subquery = (MeasureResult
                    .select(MeasureResult.id)
                    .join(MeasurePerSection)
                    .join(Measure)
                    .where(Measure.measure_type_id == 5))
    else:
        raise ValueError("Measure type not found")

    _query = (MeasureResultSection
              .update(cost=MeasureResultSection.cost * multiplier)
              .where(MeasureResultSection.measure_result_id.in_(subquery)))

    _query.execute()
    _connected_db.close()



def copy_database(vr_config: VrtoolConfig, suffix: str):
    """Copy a database and rename it with a suffix. Also reassign the input_database_name in the vr_config"""
    database_name = vr_config.input_database_name.strip(".db")
    new_name = database_name + f"_{suffix}.db"
    shutil.copy(vr_config.input_database_path, vr_config.input_directory.joinpath(new_name))
    vr_config.input_database_name = new_name


_input_model = Path(
    r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\41-1_test_automation\automation")
assert _input_model.exists()
_vr_config = VrtoolConfig().from_json(Path(_input_model).joinpath("config.json"))
_vr_config.input_directory = _input_model

copy_database(_vr_config, "modified_stability")
# modify_cost_measure_database(_vr_config, multiplier=2, measure_type="soil reinforcement")
modify_initial_beta_stability(_vr_config, std_beta=1.0)
rerun_database(_vr_config, rerun_all=True)
run_dsn_lenient_and_stringent(_vr_config, run_strict=False)



