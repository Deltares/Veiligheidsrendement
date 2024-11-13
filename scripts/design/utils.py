import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.postprocessing.database_access_functions import get_optimization_steps_for_run_id, get_measures_for_run_id, \
    import_original_assessment
from scripts.postprocessing.database_analytics import get_minimal_tc_step, get_measures_per_step_number, \
    get_reliability_for_each_step, assessment_for_each_step, calculate_traject_probability_for_steps
from vrtool.common.enums import MechanismEnum
import copy

from vrtool.orm.models import MeasureResultMechanism, MechanismPerSection, Mechanism, MeasureResultSection, \
    MeasureResult, MeasurePerSection, Measure, MeasureType, DikeTrajectInfo
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta, beta_to_pf


def get_traject_probs(db_path: Path, has_revetment: bool = False) -> list[tuple[tuple[int], list[float]]]:
    """
    Return a list of the traject probabilities for each step in the optimization process.
    Args:
        db_path:

    Returns:
    list of tuples, where each tuple contains a tuple of years and a list of probabilities
    example item: ((0, 19, 20, 25, 50, 75, 100),
  [0.09562056040494016,
   0.09606587510969489,
   0.09609719749741086,
   0.09626876542106155,
   0.09764883134543301,
   0.10075928830016645,
   0.10811136593394344])
    """

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

    return traject_probs

def calculate_traject_probability(traject_prob):
    p_nonf = [1] * len(list(traject_prob.values())[0].values())
    for mechanism, data in traject_prob.items():
        time, pf = zip(*sorted(data.items()))

        p_nonf = np.multiply(p_nonf, np.subtract(1, pf))
    return time, list(1 - p_nonf)



def get_measures_for_all_sections(db_path, t_design):
    # Fetch all MeasureResultMechanism records for the specified year
    measures_for_all_sections_beta = (MeasureResultMechanism
                                      .select(MeasureResultMechanism, MechanismPerSection, Mechanism.name)
                                      .join(MechanismPerSection, on=(
            MechanismPerSection.id == MeasureResultMechanism.mechanism_per_section_id))
                                      .join(Mechanism, on=(Mechanism.id == MechanismPerSection.mechanism_id))
                                      .where(MeasureResultMechanism.time == t_design))

    # Fetch all MeasureResultSection records for the specified year
    measures_for_all_sections_cost = (MeasureResultSection
                                      .select()
                                      .where(MeasureResultSection.time == t_design))

    measure_types_for_all_sections = (MeasureResult
                                      .select(MeasureResult, MeasurePerSection, Measure, MeasureType)
                                      .join(MeasurePerSection,
                                            on=(MeasurePerSection.id == MeasureResult.measure_per_section_id))
                                      .join(Measure, on=(Measure.id == MeasurePerSection.measure_id))
                                      .join(MeasureType, on=(MeasureType.id == Measure.measure_type_id)))

    # Convert the beta query results to a list of dictionaries
    beta_data_list = []
    for measure in measures_for_all_sections_beta:
        data = measure.__data__
        data['mechanism_name'] = measure.mechanism_per_section.mechanism.name
        data['section_id'] = measure.mechanism_per_section.section_id
        beta_data_list.append(data)

    measure_type_data_list = []
    for measure in measure_types_for_all_sections:
        data = measure.__data__
        data['measure_type_id'] = measure.measure_per_section.measure.measure_type.id
        measure_type_data_list.append(data)

    # Convert to Pandas DataFrame
    beta_df = pd.DataFrame(beta_data_list)
    beta_df.drop(columns=['id', 'mechanism_per_section'], inplace=True)
    beta_df = beta_df.pivot(index=['section_id', 'measure_result'], columns=['mechanism_name'], values='beta')
    # remove mechanism_name from the index
    beta_df.reset_index(inplace=True)
    beta_df.set_index('measure_result', inplace=True)

    # Convert the cost query results to a list of dictionaries
    cost_data_list = []
    for measure in measures_for_all_sections_cost:
        cost_data_list.append({'measure_result_id': measure.measure_result_id, 'cost': measure.cost})
    # Convert to Pandas DataFrame
    cost_df = pd.DataFrame(cost_data_list)
    cost_df.set_index('measure_result_id', inplace=True)

    # measure type list
    measure_type_data = pd.DataFrame(measure_type_data_list)[['id', 'measure_type_id']].rename(
        columns={'id': 'measure_result'})
    measure_type_data.drop_duplicates(inplace=True)
    df = beta_df.join(measure_type_data.set_index('measure_result'))
    # Join the beta and cost DataFrames
    df = df.join(cost_df)

    return df


def add_combined_vzg_soil(df):
    # copy original dataframe
    df_out = df.copy()
    for section in df.section_id.unique():
        # we get all soil reinforcements (all measures with measure_type_id 1) and make a new df of it. Make sur it is a copy
        soil = df[(df.section_id == section) & (df.measure_type_id == 1)].copy()
        vzg = df[(df.section_id == section) & (df.measure_type_id == 3)].copy()
        soil.loc[:, 'cost'] = np.add(soil.cost, vzg['cost'].values[0])

        beta_piping = soil.Piping
        new_beta_piping = pf_to_beta(beta_to_pf(beta_piping) / 1000.)
        soil.loc[:, 'Piping'] = new_beta_piping
        soil.loc[:, 'measure_type_id'] = 99
        df_out = pd.concat([df_out, soil])
    return df_out


def compute_traject_probability(minimal_cost_dataset: pd.DataFrame):
    """

    Args:
        minimal_cost_dataset: DataFrame with 1 row per vak. The row contains the beta and cost for the measure with
        that minimizes the cost and fullfills the eis

    Returns:

    """
    # no upscaling in sections.
    pf_overflow = max(beta_to_pf(minimal_cost_dataset['Overflow']))
    p_nonf_piping = np.product(np.subtract(1, beta_to_pf(minimal_cost_dataset['Piping'])))
    p_nonf_stability = np.product(np.subtract(1, beta_to_pf(minimal_cost_dataset['StabilityInner'])))
    pf_traject = 1 - (1 - pf_overflow) * (p_nonf_piping) * (p_nonf_stability)
    return pf_traject


def calculate_cost(overflow_beta, piping_beta, stability_beta, measures_df, correct_LE=False) -> tuple[float, float]:
    """
    return the minimal cost and the corresponding traject faalkans for the given beta values
    Args:
        overflow_beta:
        piping_beta:
        stability_beta:
        measures_df:
        correct_LE:

    Returns:

    """
    # get all sections
    sections = measures_df['section_id'].unique()

    # Keep only measures for which DSN beta are higher than VR betas
    possible_measures = measures_df.loc[(measures_df['Overflow_dsn'] >= overflow_beta) &
                                        (measures_df['Piping_dsn'] >= piping_beta) &
                                        (measures_df['StabilityInner_dsn'] >= stability_beta)]
    # get the minimal cost for each section_id and the betas that belong to that measure
    minimal_costs_idx = possible_measures.reset_index().groupby('section_id')['cost'].idxmin()
    minimal_costs_data = possible_measures.reset_index().loc[minimal_costs_idx]

    computed_traject_probability = compute_traject_probability(minimal_costs_data)

    minimal_costs = minimal_costs_data['cost']
    # check if all sections are in the minimal_costs, if any of them is not in there return 1e99, else return the sum of the costs
    if len(sections) != len(minimal_costs):
        return 1e99, computed_traject_probability
    else:
        return minimal_costs.sum(), computed_traject_probability



def get_target_beta_grid(N_omega, N_LE):
    """
    Return a iterator of (beta_overflow, beta_piping, beta_stability_inner) for every combination p_eis / ( N_omega * N_LE)


    Returns:

    """
    p_max = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).p_max * 0.52
    omega_piping = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).omega_piping
    omega_stability_inner = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).omega_stability_inner
    omega_overflow = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).omega_overflow
    a_piping = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).a_piping
    a_stability_inner = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).a_stability_inner
    b_stability_inner = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).b_stability_inner
    b_piping = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).b_piping
    traject_length = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).traject_length

    N_overflow_grid = N_omega.copy()
    N_piping_grid = sorted(set([a * b for a, b in list(itertools.product(N_omega, N_LE))]))
    N_stability_inner_grid = sorted(set([a * b for a, b in list(itertools.product(N_omega, N_LE))]))

    # add existing values:
    N_overflow_grid = N_overflow_grid + [np.divide(1, omega_overflow)]
    N_piping_grid = N_piping_grid + [np.divide(1, omega_piping) * np.divide(a_piping * traject_length, b_piping)]
    N_stability_inner_grid = N_stability_inner_grid + [
        np.divide(1, omega_stability_inner) * np.divide(a_stability_inner * traject_length, b_stability_inner)]

    # make a beta_grid for all
    overflow_grid = pf_to_beta(np.divide(p_max, N_overflow_grid))
    piping_grid = pf_to_beta(np.divide(p_max, N_piping_grid))
    stability_inner_grid = pf_to_beta(np.divide(p_max, N_stability_inner_grid))

    # #make a grid for all 3 mechanisms.
    target_beta_grid = itertools.product(overflow_grid, piping_grid, stability_inner_grid)
    print(f"Grid dimensions are {len(overflow_grid)} x {len(piping_grid)} x {len(stability_inner_grid)}")
    print(target_beta_grid)

    return target_beta_grid


def get_cost_traject_pf_combinations(target_beta_grid: pd.DataFrame, measures_df_with_dsn: pd.DataFrame) -> tuple[list, list]:
    """
    Return a list of cost and a list of traject faalkans for every combination of betas in the grid target_beta_grid

    params: target_beta_grid: list of tuple (beta_overflow, beta_piping, beta_stability) for every combination of p_eis/
    N_omega *N_LE.
    params: measures_df_with_dsn: DataFrame containing all the measures and their betas.
    """
    cost = []
    pf_traject = []
    for count, (overflow_beta, piping_beta, stability_beta) in enumerate(list(copy.deepcopy(target_beta_grid))):
        cost_i, pf_traject_i = calculate_cost(overflow_beta, piping_beta, stability_beta, measures_df_with_dsn)
        if cost_i < 1.e99:
            cost.append(cost_i)
            pf_traject.append(pf_traject_i)
        else:
            pass
            # print(f"Skipping beta combination {overflow_beta, piping_beta, stability_beta}")
    return cost, pf_traject