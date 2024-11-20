import itertools
from pathlib import Path
import copy

import numpy as np
import pandas as pd

from scripts.design.plot_design import plot_combined_measure_figure, plot_sensitivity, plot_sensitivity_plotly, \
    plot_histogram_metrics
from scripts.design.utils import get_traject_probs, get_target_beta_grid, get_cost_traject_pf_combinations, \
    get_measures_df_with_dsn, get_dsn_point_pf_cost, get_least_expensive_combination_point, get_vr_eco_optimum_point
from scripts.postprocessing.database_access_functions import get_overview_of_runs, get_optimization_steps_for_run_id
from scripts.postprocessing.database_analytics import get_minimal_tc_step

from vrtool.orm.models import DikeTrajectInfo


def run_single_database(db_path: Path, plot: bool = False):
    # Input
    has_revetment = False
    LE = False
    t_design = 50

    res_run = {
        "cheapest_combination_cost": None,
        "cheapest_combination_pf": None,
        "dsn_point_cost": None,
        "dsn_point_pf": None,
        "vrm_eco_point_cost": None,
        "vrm_eco_point_pf": None,
        "vrm_optimal_point_cost": None,
        "vrm_optimal_point_pf": None,
        "N_LE": None,
        "N_omega": None,

    }
    traject_probs = get_traject_probs(db_path)
    p_max = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).p_max * 0.52
    traject_name = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).traject_name
    optimization_steps = get_optimization_steps_for_run_id(db_path, 1)
    ind_2075 = np.where(np.array(traject_probs[0][0]) == 50)[0][0]  # find index where traject_probs[0][0] == 50
    pf_2075 = [traject_probs[i][1][ind_2075] for i in range(len(traject_probs))]
    cost_vrm = [optimization_steps[i]['total_lcc'] for i in range(len(traject_probs))]

    #### Get all combinations

    N_omega = [2., 4., 8., 16., 32.]
    N_LE = [5., 10., 20., 40., 50.]
    combination_df = get_target_beta_grid(N_omega, N_LE)
    # Put all the measures in a DataFrame for DSN
    measures_df_with_dsn = get_measures_df_with_dsn(LE, t_design)
    df_combinations_results = get_cost_traject_pf_combinations(combination_df, measures_df_with_dsn)
    cheapest_combination = get_least_expensive_combination_point(df_combinations_results, p_max)

    res_run["cheapest_combination_cost"] = cheapest_combination[0]
    res_run["cheapest_combination_pf"] = cheapest_combination[1]
    res_run["dsn_point_cost"] = get_dsn_point_pf_cost(db_path)[0]
    res_run["dsn_point_pf"] = get_dsn_point_pf_cost(db_path)[1]
    res_run["vrm_eco_point_cost"] = get_vr_eco_optimum_point(traject_probs, optimization_steps)[0]
    res_run["vrm_eco_point_pf"] = get_vr_eco_optimum_point(traject_probs, optimization_steps)[1]

    # Get distance between VR path and the least expensive combination
    # Point 1: VR lowest complying with the target reliability
    step_idx_pf_2075 = np.argwhere(np.array(pf_2075) < p_max * .52)[0][0]
    res_run["vrm_optimal_point_cost"] = cost_vrm[step_idx_pf_2075]
    res_run["vrm_optimal_point_pf"] = pf_2075[step_idx_pf_2075]

    # path VRM
    vrm_optimization_steps = pd.DataFrame({"cost": cost_vrm, "pf_traject": pf_2075})

    if plot:
        plot_combined_measure_figure(df_combinations=df_combinations_results,
                                     vrm_optimization_steps=vrm_optimization_steps,
                                     vrm_optimum_point=get_vr_eco_optimum_point(traject_probs, optimization_steps),
                                     least_expensive_combination=cheapest_combination,
                                     dsn_point=get_dsn_point_pf_cost(db_path),
                                     p_max=p_max)

    return res_run, vrm_optimization_steps, df_combinations_results, p_max
