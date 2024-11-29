from pathlib import Path

import numpy as np

from scripts.design.run_vrtool_specific import run_dsn_lenient_and_stringent, rerun_database
from vrtool.api import ApiRunWorkflows
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.models import MeasureResultSection, MeasureResult, MeasurePerSection, Measure, \
    ComputationScenarioParameter, MeasureResultMechanism, Mechanism, MechanismPerSection
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


def modify_beta_measure_database(_vr_config: VrtoolConfig, measure_type: str, mechanism: str, std_beta: float):
    """
    Modify the beta for a specific mechanism in the measure database.

    Args:
        _vr_config:
        measure_type:
        mechanism: one of "Overflow", "Piping", "StabilityInner", "Revetment". Check for spelling.
        std_beta:

    """
    # Open the database and reduce cost of the measures:
    _connected_db = open_database(_vr_config.input_database_path)

    subquery_measure_result = get_subquery_for_measure_type(measure_type)

    subquery_mechanism_result = get_subquery_for_mechanism(mechanism)

    _query = (MeasureResultMechanism
    .update(beta=MeasureResultMechanism.beta + np.random.normal(0, std_beta))
    .where(
        MeasureResultMechanism.measure_result_id.in_(subquery_measure_result) &
        MeasureResultMechanism.mechanism_per_section_id.in_(subquery_mechanism_result)
    ))

    _query.execute()
    _connected_db.close()


def modify_cost_measure_database(_vr_config: VrtoolConfig, multiplier: int = 2, measure_type: str = None):
    """

    Args:
        _vr_config:
        multiplier:
        measure_type: One of "soil reinforcement", "vertical piping solution", "diaphram wall", "stability screening", "soil reinforcement + screen"

    Returns:

    """
    # Open the database and reduce cost of the measures:
    _connected_db = open_database(_vr_config.input_database_path)


    if measure_type  == "soil reinforcement":
        # apply new cost to the MeasureType == 1 directly.
        subquery_measure_result_soil_reinf = get_subquery_for_measure_type("soil reinforcement")
        _query = (MeasureResultSection
                  .update(cost=MeasureResultSection.cost * multiplier)
                  .where(MeasureResultSection.measure_result_id.in_(subquery_measure_result_soil_reinf)))


        # For MeasureType =2, we need to subtract the cost of a screen: this comes from subquery_measure_result_screen
        subquery_measure_soil_reinforcement_and_screen = get_subquery_for_measure_type("soil reinforcement + screen")
        subquery_measure_result_screen = get_subquery_for_measure_type("stability screening")
        sub = (MeasureResultSection.select().where(MeasureResultSection.measure_result_id.in_(subquery_measure_result_screen)))
        # len = 840 = 120 * 7 = 60 section * 2 Lscreen * 7 times OK


        query = (MeasureResultSection.select().where(MeasureResultSection.measure_result_id.in_(subquery_measure_soil_reinforcement_and_screen)))
        #dberm: 0/2/5/8/10/15/20/30
        #dcrest: 0/0.25/0.5/0.75/1/1.25/1.5/1.75/2
        i = 0
        for row in query: #len 60480 =  72 * 840 = 8dberm * 9dcrest * 840
            cost_screen  = sub[i // 72].cost # for every 72 consecutive rows, the cost of the screen is the same
            new_cost = (row.cost - cost_screen) * multiplier + cost_screen
            row.cost = new_cost
            row.save()
            i += 1

    else:
        subquery_measure_result = get_subquery_for_measure_type(measure_type)

        _query = (MeasureResultSection
                  .update(cost=MeasureResultSection.cost * multiplier)
                  .where(MeasureResultSection.measure_result_id.in_(subquery_measure_result)))

    _query.execute()
    _connected_db.close()


def get_subquery_for_mechanism(mechanism: str):
    """
    Get the subquery for a specific mechanism
    Args:
        mechanism: one of "Overflow", "Piping", "StabilityInner", "Revetment"

    Returns:

    """
    if mechanism == "Overflow":
        meca_id = 1
    elif mechanism == "Piping":
        meca_id = 2
    elif mechanism == "StabilityInner":
        meca_id = 3
    elif mechanism == "Revetment":
        meca_id = 4
    else:
        raise ValueError("Mechanism not found")

    subquery_mechanism_result = (MechanismPerSection.select(MechanismPerSection.id)
                                 .where(MechanismPerSection.mechanism_id == meca_id))
    return subquery_mechanism_result

def get_subquery_for_measure_type(measure_type: str):
    if measure_type == "soil reinforcement":
        subquery = (MeasureResult
                    .select(MeasureResult.id)
                    .join(MeasurePerSection)
                    .join(Measure)
                    .where(Measure.measure_type_id.in_([1])))
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
    elif measure_type == "soil reinforcement + screen":
        subquery = (MeasureResult
                    .select(MeasureResult.id)
                    .join(MeasurePerSection)
                    .join(Measure)
                    .where(Measure.measure_type_id.in_([2])))

    else:
        raise ValueError("Measure type not found")
    return subquery


def copy_database(vr_config: VrtoolConfig, suffix: str):
    """Copy a database and rename it with a suffix. Also reassign the input_database_name in the vr_config"""
    database_name = vr_config.input_database_name.strip(".db")
    new_name = database_name + f"_{suffix}.db"
    shutil.copy(vr_config.input_database_path, vr_config.input_directory.joinpath(new_name))
    vr_config.input_database_name = new_name
