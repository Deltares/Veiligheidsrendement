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
    MeasureResult, MeasurePerSection, Measure, MeasureType, DikeTrajectInfo, SectionData
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta, beta_to_pf


def get_traject_probs(db_path: Path, has_revetment: bool = False, run_id: int = 1) -> list[
    tuple[tuple[int], list[float]]]:
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

    lists_of_measures = get_measures_for_run_id(db_path, run_id)
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


def get_measures_for_all_sections(t_design):
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

def get_measures_df_with_dsn(LE: bool = False, t_design: int = 50):
    measures_df_db = get_measures_for_all_sections(t_design)
    measures_df = add_combined_vzg_soil(measures_df_db)
    # sort measures_df by section_id and measure_result
    measures_df.sort_values(by=['section_id', 'measure_result'], inplace=True)

    # as probabilities in the df are to be interpreted as probabilities per section, we need to also compute the corresponding cross-section probabilities

    # get the section lengths from the database
    section_lengths = pd.DataFrame([(section.id, section.section_length) for section in SectionData.select()],
                                   columns=['section_id', 'section_length'])
    section_lengths.set_index('section_id', inplace=True)
    # merge lengths to measures_df
    measures_df_with_dsn = measures_df.merge(section_lengths, left_on='section_id', right_index=True)
    measures_df_with_dsn['Overflow_dsn'] = measures_df_with_dsn['Overflow']
    if LE:
        N_piping = measures_df_with_dsn['section_length'] / 300
        N_piping = N_piping.apply(lambda x: max(x, 1))
        measures_df_with_dsn['Piping_dsn'] = pf_to_beta(beta_to_pf(measures_df_with_dsn['Piping']) / N_piping)
        N_stability = measures_df_with_dsn['section_length'] / 50
        N_stability = N_stability.apply(lambda x: max(x, 1))
        measures_df_with_dsn['StabilityInner_dsn'] = pf_to_beta(
            beta_to_pf(measures_df_with_dsn['StabilityInner']) / N_stability)
    else:
        measures_df_with_dsn['Piping_dsn'] = measures_df_with_dsn['Piping']
        measures_df_with_dsn['StabilityInner_dsn'] = measures_df_with_dsn['StabilityInner']
    return measures_df_with_dsn


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
    Also return the DataFrame with all the combinations of N_omega and N_LE


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

    # make a DataFrame with all the combinations of N_grid

    # combinations_df = pd.DataFrame(list(itertools.product(N_overflow_grid, N_piping_grid, N_stability_inner_grid)), columns=['N_overflow_grid', 'N_piping_grid', 'N_stability_inner_grid'])
    # # add the corresponding N_omega and N_LE
    combinations = []
    for N_overflow, N_piping, N_stability_inner in itertools.product(N_overflow_grid, N_piping_grid,
                                                                     N_stability_inner_grid):
        # Find closest matching N_omega and N_LE values for each parameter
        N_overflow_origin = min(N_omega, key=lambda x: abs(x - N_overflow))  # Closest match in N_omega
        N_piping_origin = min(N_omega, key=lambda x: abs(x - N_piping / N_LE[0]))  # Closest match in N_omega for piping
        N_stability_inner_origin = min(N_omega, key=lambda x: abs(
            x - N_stability_inner / N_LE[0]))  # Closest match in N_omega for stability

        # Add a row to the list with original values
        combinations.append((N_overflow, N_piping, N_stability_inner, N_overflow_origin, N_LE[0]))

    # Create DataFrame with mapped N_omega and N_LE columns
    combinations_df = pd.DataFrame(
        combinations,
        columns=['N_overflow_grid', 'N_piping_grid', 'N_stability_inner_grid', 'N_omega', 'N_LE']
    )

    combinations_df["overflow_grid"] = pf_to_beta(np.divide(p_max, combinations_df["N_overflow_grid"]))
    combinations_df["piping_grid"] = pf_to_beta(np.divide(p_max, combinations_df["N_piping_grid"]))
    combinations_df["stability_inner_grid"] = pf_to_beta(np.divide(p_max, combinations_df["N_stability_inner_grid"]))

    return combinations_df


def get_cost_traject_pf_combinations(combination_df: pd.DataFrame, measures_df_with_dsn: pd.DataFrame) -> pd.DataFrame:
    """
    Add the cost and the traject probability to the combination_df

    params: target_beta_grid: combination for all N_omega and N_LE, it contains also the corresponding N_stability_inner,
     N_piping, N_overflow, and the corresponding beta values
    params: measures_df_with_dsn: DataFrame containing all the measures and their betas.
    """
    cost = []
    pf_traject = []

    for _, row in combination_df.iterrows():
        cost_i, pf_traject_i = calculate_cost(row['overflow_grid'], row['piping_grid'], row['stability_inner_grid'],
                                              measures_df_with_dsn)
        if cost_i < 1.e99:
            cost.append(cost_i)
            pf_traject.append(pf_traject_i)
        else:
            cost.append(np.nan)
            pf_traject.append(np.nan)
            # print(f"Skipping beta combination {overflow_beta, piping_beta, stability_beta}")
    combination_df['cost'] = cost
    combination_df['pf_traject'] = pf_traject
    return combination_df



def get_dsn_point_pf_cost(db_path):
    """Return (cost, traject_pf) for the DSN point"""
    dsn_steps = get_optimization_steps_for_run_id(db_path, 2)
    traject_probs_dsn = get_traject_probs(db_path, run_id=2)


    ind_2075 = np.where(np.array(traject_probs_dsn[0][0]) == 50)[0][0]
    dsn_cost = dsn_steps[-1]['total_lcc']
    dsn_pf = traject_probs_dsn[-1][1][ind_2075]
    return dsn_cost, dsn_pf

def get_vr_eco_optimum_point(traject_probs, optimization_steps):

    considered_tc_step = get_minimal_tc_step(optimization_steps) - 1
    # find index where traject_probs[0][0] == 50
    ind_2075 = np.where(np.array(traject_probs[0][0]) == 50)[0][0]
    pf_2075 = [traject_probs[i][1][ind_2075] for i in range(len(traject_probs))]
    cost_vrm = [optimization_steps[i]['total_lcc'] for i in range(len(traject_probs))]

    vrm_optimum_cost = optimization_steps[considered_tc_step - 1]['total_lcc']
    vrm_optimum_pf = traject_probs[considered_tc_step - 1][1][ind_2075]
    return vrm_optimum_cost, vrm_optimum_pf

def get_least_expensive_combination_point(df_combinations_results, p_max):
    cost = np.array(df_combinations_results['cost'])
    pf_traject = np.array(df_combinations_results['pf_traject'])
    # find the measure with the lowest cost that is compliant with the p_max
    df = pd.DataFrame({'cost': cost, 'pf': pf_traject})
    df = df[df['pf'] <= p_max]
    min_cost = df['cost'].min()
    min_cost_idx = df[df['cost'] == min_cost].index[0]
    return min_cost, pf_traject[min_cost_idx]