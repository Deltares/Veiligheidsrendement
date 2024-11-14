import copy
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import norm

from scripts.postprocessing.database_access_functions import get_overview_of_runs, get_optimization_steps_for_run_id, \
    get_measures_for_run_id, import_original_assessment
from scripts.postprocessing.database_analytics import get_minimal_tc_step, get_measures_per_step_number, \
    get_reliability_for_each_step, assessment_for_each_step, calculate_traject_probability_for_steps
from vrtool.common.enums import MechanismEnum


def calculate_traject_probability(traject_prob):
    p_nonf = [1] * len(list(traject_prob.values())[0].values())
    for mechanism, data in traject_prob.items():
        time, pf = zip(*sorted(data.items()))
        p_nonf = np.multiply(p_nonf, np.subtract(1, pf))
    return time, list(1 - p_nonf)

def get_probability_for_mechanism_in_year_per_section(assessment_step: dict[int:dict[str:list]], year: int,
                                                      mechanism: MechanismEnum):
    probability_per_section = {}
    time_index = np.argwhere(np.array(assessment_step[mechanism][1]['time']) == year).flatten()[0]
    for section in assessment_step[mechanism]:
        probability_per_section[section] = assessment_step[mechanism][section]['beta'][time_index]
    return probability_per_section

def get_vr_eis_index_2075(traject_config: dict) -> int:
    db_path = traject_config['path']
    has_revetment = traject_config['revetment']
    eis = traject_config["eis"]

    _runs_overview = get_overview_of_runs(db_path)

    optimization_steps = get_optimization_steps_for_run_id(db_path, 1)
    considered_tc_step = get_minimal_tc_step(optimization_steps) - 1

    lists_of_measures = get_measures_for_run_id(db_path, 1)
    measures_per_step = get_measures_per_step_number(lists_of_measures)

    assessment_results = {}
    for mechanism in [MechanismEnum.OVERFLOW, MechanismEnum.PIPING, MechanismEnum.STABILITY_INNER,
                      MechanismEnum.REVETMENT]:
        if has_revetment or mechanism != MechanismEnum.REVETMENT:
            assessment_results[mechanism] = import_original_assessment(db_path, mechanism)

    reliability_per_step = get_reliability_for_each_step(db_path, measures_per_step)

    stepwise_assessment = assessment_for_each_step(copy.deepcopy(assessment_results), reliability_per_step)

    traject_probability = calculate_traject_probability_for_steps(stepwise_assessment)

    traject_probs = [calculate_traject_probability(traject_probability_step) for traject_probability_step in
                     traject_probability]

    pf_2075 = [traject_probs[i][1][4] for i in
               range(len(traject_probs))]  # list of traject faalkans at 2075 for every step

    if traject_config["meet_eis"]:
        # most trajects have an optimization step where pf<eis*.52 but sometimes not
        step_idx_pf_2075 = np.argwhere(np.array(pf_2075) < eis * .52)[0][0]

    else:
        step_idx_pf_2075 = len(pf_2075) - 1  # fall back in case the threshold is not reached
    return step_idx_pf_2075




def get_traject_requirements_per_sections(traject_config: dict):
    db_path = traject_config['path']
    has_revetment = traject_config['revetment']
    eis = traject_config["eis"]

    _runs_overview = get_overview_of_runs(db_path)

    optimization_steps = get_optimization_steps_for_run_id(db_path, 1)
    considered_tc_step = get_minimal_tc_step(optimization_steps) - 1

    lists_of_measures = get_measures_for_run_id(db_path, 1)
    measures_per_step = get_measures_per_step_number(lists_of_measures)

    assessment_results = {}
    for mechanism in [MechanismEnum.OVERFLOW, MechanismEnum.PIPING, MechanismEnum.STABILITY_INNER,
                      MechanismEnum.REVETMENT]:
        if has_revetment or mechanism != MechanismEnum.REVETMENT:
            assessment_results[mechanism] = import_original_assessment(db_path, mechanism)

    reliability_per_step = get_reliability_for_each_step(db_path, measures_per_step)

    stepwise_assessment = assessment_for_each_step(copy.deepcopy(assessment_results), reliability_per_step)


    step_idx_pf_2075 = get_vr_eis_index_2075(traject_config)
    gehanteerde_stap = step_idx_pf_2075  # 329 is optimaal
    # gehanteerde_stap = considered_tc_step  #329 is optimaal, 260 is conform norm
    beschouwd_jaar = 50

    # this is the beta of EVERY SECTION in 2075 at the selected step (260 here which is conform norm) for every mechanism
    requirements_per_section = {mechanism: get_probability_for_mechanism_in_year_per_section(
        assessment_step=stepwise_assessment[gehanteerde_stap],
        year=beschouwd_jaar,
        mechanism=mechanism)
        for mechanism in
        [MechanismEnum.OVERFLOW, MechanismEnum.PIPING, MechanismEnum.STABILITY_INNER]}

    return requirements_per_section


def calculate_strick_boundaries(val):
    if val > 1000:
        return 1000
    elif val <= 1000:
        if val <= 300:
            if val <= 100:
                if val <= 30:
                    return val
                else:
                    return 100
            else:
                return 300
        else:
            return 1000


def calculate_lenient_boundaries(val):
    if val > 1000:
        return 1000
    elif val <= 1000:
        if val <= 300:
            if val <= 100:
                if val <= 30:
                    return val
                else:
                    return 30
            else:
                return 100
        else:
            return 300


def create_requirement_files(requirements_per_section: dict, trajec_config: dict):
    requirements_df = pd.DataFrame.from_dict(requirements_per_section)

    requirements_df["pf_OVERFLOW"] = norm.cdf(-requirements_df[MechanismEnum.OVERFLOW])
    requirements_df["pf_PIPING"] = norm.cdf(-requirements_df[MechanismEnum.PIPING])
    requirements_df["pf_STABILITY_INNER"] = norm.cdf(-requirements_df[MechanismEnum.STABILITY_INNER])

    # ratios
    requirements_df["ratio_OVERFLOW"] = trajec_config["eis"] / requirements_df["pf_OVERFLOW"]
    requirements_df["ratio_PIPING"] = trajec_config["eis"] / requirements_df["pf_PIPING"]
    requirements_df["ratio_STABILITY_INNER"] = trajec_config["eis"] / requirements_df["pf_STABILITY_INNER"]

    min_overflow = requirements_df["ratio_OVERFLOW"].min()

    # strick requirements
    # overflow: assign a constant equal to 1.4
    requirements_df["strick_overflow"] = min_overflow
    requirements_df["strick_piping"] = requirements_df["ratio_PIPING"].apply(calculate_strick_boundaries)
    requirements_df["strick_stability_inner"] = requirements_df["ratio_STABILITY_INNER"].apply(
        calculate_strick_boundaries)

    strick_df = pd.DataFrame(columns=["section_name", "OVERFLOW", "PIPING", "STABILITY_INNER"])
    strick_df.reset_index(drop=True, inplace=True)

    strick_df["section_name"] = requirements_df.index
    strick_df.index += 1  # start index at 1
    strick_df["OVERFLOW"] = requirements_df["strick_overflow"]
    strick_df["PIPING"] = requirements_df["strick_piping"]
    strick_df["STABILITY_INNER"] = requirements_df["strick_stability_inner"]
    strick_df.to_csv(trajec_config['path'].parent.joinpath('requirements_strict.csv'), sep=",", index=False)

    # lenient requirements
    requirements_df["lenient_overflow"] = min_overflow
    requirements_df["lenient_piping"] = requirements_df["ratio_PIPING"].apply(calculate_lenient_boundaries)
    requirements_df["lenient_stability_inner"] = requirements_df["ratio_STABILITY_INNER"].apply(
        calculate_lenient_boundaries)
    lenient_df = pd.DataFrame(columns=["section_name", "OVERFLOW", "PIPING", "STABILITY_INNER"])
    lenient_df["section_name"] = requirements_df.index
    lenient_df.index += 1  # start index at 1
    lenient_df["OVERFLOW"] = requirements_df["lenient_overflow"]
    lenient_df["PIPING"] = requirements_df["lenient_piping"]
    lenient_df["STABILITY_INNER"] = requirements_df["lenient_stability_inner"]
    lenient_df.to_csv(trajec_config['path'].parent.joinpath('requirements_lenient.csv'), sep=",", index=False)


traject_config = {
    "name": "41-1",
    "path": Path(
        # r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\41-1_test_automation\ref\41-1_database_origineel.db"
        # r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\41-1_test_automation\expensive\41-1_database_origineel_modified_expensive.db"
        r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\41-1_test_automation\cheap\41-1_database_origineel_cheap.db"
    ),
    "eis": 1 / 10000,
    "l_traject": 12500,
    "revetment": False,
    "meet_eis": True,
}

requirements_per_section = get_traject_requirements_per_sections(traject_config)
create_requirement_files(requirements_per_section, traject_config)
