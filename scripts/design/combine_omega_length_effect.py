import itertools
from pathlib import Path
import copy

import numpy as np
import pandas as pd

from scripts.design.plot_design import plot_combined_measure_figure
from scripts.design.utils import get_target_beta_grid, get_cost_traject_pf_combinations, \
    get_measures_df_with_dsn, get_dsn_point_pf_cost, get_least_expensive_combination_point, get_vr_eco_optimum_point
from scripts.design.vrtool_optimization_object import VRTOOLOptimizationObject

from vrtool.orm.models import DikeTrajectInfo


def create_summary_results_csv(dir_path: Path, filename: str, run_id: int = 1):
    """
    Go through all databases in a direction and apply combination of N_omega and N_LE to get:
        - the cheapest combination complying with eis
        - the DSN point
        - the VRM eco point
        - the VRM optimal point (lowest cost complying with the target reliability on the VR pad)

    Args:
        dir_path:
        filename: save result csv
        run_id: some database have the correct run_id on 1 or 3.

    Returns:

    """
    res_list = []
    for idx, db_path in enumerate(dir_path.glob("*.db")):

        if idx == 0:
            run_idd = 1  # force run_id for base case
            assert db_path.stem == "atabase_10-3.sqlite_0", "Verify that first iterated database is the base case"
            # assert db_path.stem == "38-1_basis_0", "Verify that first iterated database is the base case"
        else:
            run_idd = run_id

        print(db_path, "processing ...", run_idd)
        res, vrm_optimization_steps, df_combinations_results, p_max = run_combination_single_database(db_path,
                                                                                                      plot=False,
                                                                                                      run_id=run_idd)
        res_list.append(res)

    df = pd.DataFrame(res_list)
    df.to_csv(dir_path.joinpath(f"{filename}.csv"))


def run_combination_single_database(db_path: Path, plot: bool = False, run_id: int = 1):
    vrm_run = VRTOOLOptimizationObject(db_path, run_id)
    vrm_run.get_all_optimization_results()
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
        "vrm_ondergrenz_point_cost": None,
        "vrm_ondergrenz_point_pf": None,
        "N_LE": None,
        "N_omega": None,
    }

    p_max = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).p_max * 0.52
    traject_name = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).traject_name
    optimization_steps = vrm_run.optimization_steps
    traject_probs = vrm_run.traject_probs
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

    dsn_point_cost, dsn_point_pf = get_dsn_point_pf_cost(db_path, run_id_dsn=run_id + 1)
    res_run["cheapest_combination_cost"] = cheapest_combination[0]
    res_run["cheapest_combination_pf"] = cheapest_combination[1]
    res_run["dsn_point_cost"] = dsn_point_cost
    res_run["dsn_point_pf"] = dsn_point_pf
    res_run["vrm_eco_point_cost"] = get_vr_eco_optimum_point(traject_probs, optimization_steps)[0]
    res_run["vrm_eco_point_pf"] = get_vr_eco_optimum_point(traject_probs, optimization_steps)[1]

    # Get distance between VR path and the least expensive combination
    # Point 1: VR lowest complying with the target reliability
    step_idx_pf_2075 = np.argwhere(np.array(pf_2075) < p_max)[0][0]
    res_run["vrm_ondergrenz_point_cost"] = cost_vrm[step_idx_pf_2075]
    res_run["vrm_ondergrenz_point_pf"] = pf_2075[step_idx_pf_2075]

    # path VRM
    vrm_optimization_steps = pd.DataFrame({"cost": cost_vrm, "pf_traject": pf_2075})

    if plot:
        fig = plot_combined_measure_figure(df_combinations=df_combinations_results,
                                           vrm_optimization_steps=vrm_optimization_steps,
                                           vrm_optimum_point=get_vr_eco_optimum_point(traject_probs,
                                                                                      optimization_steps),
                                           least_expensive_combination=cheapest_combination,
                                           dsn_point=(dsn_point_cost, dsn_point_pf),
                                           p_max=p_max,
                                           N_color_mode=True)
        save_dir = db_path.parent.joinpath("bolletje_figuur")
        # fig.savefig(save_dir.joinpath(f"{traject_name}_{db_path.stem}.png"), dpi=300)
    return res_run, vrm_optimization_steps, df_combinations_results, p_max
