from cProfile import label
from pathlib import Path

import pandas as pd
import seaborn as sns

from scripts.design.create_requirement_files import get_vr_eis_index_2075, traject_config

# sns.set(style="whitegrid")
# colors = sns.color_palette("colorblind", 10)
import matplotlib.pyplot as plt

from scripts.postprocessing.database_access_functions import get_optimization_steps_for_run_id
from scripts.postprocessing.database_analytics import get_minimal_tc_step
from scripts.postprocessing.generate_output import plot_lcc_tc_from_steps


def plot_1():
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
    results_dir = traject_config['path'].parent
    database_name = traject_config['path'].name

    step_idx_pf_2075 = get_vr_eis_index_2075(traject_config)

    databases = {'Veiligheidsrendement': (database_name, 3),
                 'Default doorsnede-eisen': (database_name, 4),
                 # 'Herverdeling omega' : (database_name,6),
                 'Vakspecifiek soepel': (database_name, 5),
                 'Vakspecifiek streng': (database_name, 6),
                 }

    optimization_steps_vakspecifiek = {case: get_optimization_steps_for_run_id(results_dir.joinpath(database), run_id)
                                       for case, (database, run_id) in databases.items()}
    considered_tc_step_vakspecifiek = {case: get_minimal_tc_step(optimization_steps_vakspecifiek) - 1 for
                                       case, optimization_steps_vakspecifiek in optimization_steps_vakspecifiek.items()}

    fig, ax = plt.subplots()
    markers = ['d', 'o', 's', 'v', '^']
    for count, run in enumerate(optimization_steps_vakspecifiek.keys()):
        plot_lcc_tc_from_steps(optimization_steps_vakspecifiek[run], axis=ax, lbl=run, clr=colors[count])
        ax.plot(optimization_steps_vakspecifiek[run][considered_tc_step_vakspecifiek[run]]['total_lcc'],
                optimization_steps_vakspecifiek[run][considered_tc_step_vakspecifiek[run]]['total_risk'],
                markers[count], color=colors[count])

        # add the point for 1/1000 * .52 if veilgiheidsrendement
        if run == 'Veiligheidsrendement':
            ax.plot(optimization_steps_vakspecifiek[run][step_idx_pf_2075]['total_lcc'],
                    optimization_steps_vakspecifiek[run][step_idx_pf_2075]['total_risk'], markers[count],
                    color=colors[count])

    # ax.plot([1e7, 5e7,1e8, 2e8,3e8,4e8,5e8],[1e7, 5e7,1e8,2e8,3e8,4e8, 5e8], color="black")
    ax.set_xlabel('Totaal LCC')
    ax.set_ylabel('Totaal overstromingsrisico')
    ax.set_yscale('log')
    ax.set_xlim(left=0)
    ax.set_ylim(top=1e10)
    ax.legend()
    plt.show()


def plot_combined_measure_figure(
        df_combinations: pd.DataFrame,
        cost_vrm: list,
        pf_2075_vrm: list,
        cost_vrm_filtered: list,
        pf_2075_vrm_filtered: list,
        vrm_optimum_point: tuple[float, float],
        least_expensive_combination: tuple[float, float],
        dsn_point: tuple[float, float],

        p_max: float):
    from scripts.design.deltares_colors import colors

    fig, ax = plt.subplots()
    ax.scatter(df_combinations['cost'], df_combinations['pf_traject'], color=colors[4], marker='.')
    ax.plot(dsn_point[0], dsn_point[1], color=colors[6], marker='o', linestyle='', label='DSN')
    ax.plot(cost_vrm, pf_2075_vrm, color=colors[2], linestyle=':', label='Optimalisatiestappen')
    ax.plot(cost_vrm_filtered, pf_2075_vrm_filtered, color=colors[0], label='Optimalisatiestappen gefilterd')
    ax.plot(vrm_optimum_point[0], vrm_optimum_point[1], marker='o', color=colors[0], label='VRM optimum')
    ax.plot(least_expensive_combination[0], least_expensive_combination[1], marker='o', color=colors[7],
            label='Combi optimum')
    # take maximum of the cost and set the x-axis limit
    ax.set_xlim(left=0, right=df_combinations['cost'].max())
    ax.hlines(p_max, 0, df_combinations['cost'].max(), colors='k', linestyles='dashed', label='Ondergrens')
    ax.set_ylim(top=p_max * 10, bottom=p_max / 10)
    ax.set_xlabel('Kosten (M€)')
    ax.set_ylabel('Traject faalkans in 2075')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    # get xtick labels and divide by 1e6 and replace
    ax.set_xticklabels([f'{x / 1e6:.0f}' for x in ax.get_xticks()])
    ax.grid(True, which='both', linestyle=':')
    
    # save the figure
    save_dir = Path(r'C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\41-1_test_automation')
    # plt.savefig(save_dir.joinpath('38-1_geenLE_smaller_grid.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_sensitivity(df_sensitivity: pd.DataFrame):
    fig, ax = plt.subplots()
    from scripts.design.deltares_colors import colors

    ax.scatter(df_sensitivity["least_expensive_combination_cost"], df_sensitivity["least_expensive_combination_pf"], color=colors[4], label='Combi optimum')
    ax.scatter(df_sensitivity["dsn_point_cost"], df_sensitivity["dsn_point_pf"], color=colors[6], label='DSN')
    ax.scatter(df_sensitivity["vrm_eco_point_cost"], df_sensitivity["vrm_eco_point_pf"], color=colors[0], label='VRM optimum')
    ax.set_xlim(left=0)
    ax.set_xlabel('Kosten (M€)')
    ax.set_ylabel('Traject faalkans in 2075')
    ax.set_yscale('log')
    ax.legend()

    plt.show()
