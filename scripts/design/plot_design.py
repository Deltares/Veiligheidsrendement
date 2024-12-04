from cProfile import label
from pathlib import Path

import pandas as pd
import seaborn as sns

# import plotly.graph_objects as go
from scripts.design.create_requirement_files import get_vr_eis_index_2075

# sns.set(style="whitegrid")
# colors = sns.color_palette("colorblind", 10)
import matplotlib.pyplot as plt

from scripts.postprocessing.database_access_functions import get_optimization_steps_for_run_id
from scripts.postprocessing.database_analytics import get_minimal_tc_step
from scripts.postprocessing.generate_output import plot_lcc_tc_from_steps


def plot_combined_measure_figure(
        df_combinations: pd.DataFrame,
        vrm_optimization_steps: pd.DataFrame,
        vrm_optimum_point: tuple[float, float],
        least_expensive_combination: tuple[float, float],
        dsn_point: tuple[float, float],
        p_max: float,
        N_color_mode: bool = False,
):
    from scripts.design.deltares_colors import colors

    fig, ax = plt.subplots()

    if N_color_mode:
        df_combinations['N_overflow_grid'] = df_combinations['N_overflow_grid'].astype(str)
        df_combinations['N_piping_grid'] = df_combinations['N_piping_grid'].astype(str)
        df_combinations['N_stability_inner_grid'] = df_combinations['N_stability_inner_grid'].astype(str)

        sns.scatterplot(data=df_combinations, x="cost", y="pf_traject",
                        hue="N_overflow_grid")  # N_piping_grid N_stability_inner_grid
        color_dsn = "black"
        # give a star shape to marker
        ax.plot(dsn_point[0], dsn_point[1], color=color_dsn, marker='*', linestyle='', label='DSN', markersize=10)
        ax.plot(least_expensive_combination[0], least_expensive_combination[1], marker='D', color="black",
                label='Combi optimum')
        ax.plot(vrm_optimum_point[0], vrm_optimum_point[1], marker='o', color=colors[0], label='VRM optimum',)

    else:
        ax.scatter(df_combinations['cost'], df_combinations['pf_traject'], color=colors[4], marker='.')
        ax.plot(dsn_point[0], dsn_point[1], color=colors[6], marker='o', linestyle='', label='DSN')
        ax.plot(least_expensive_combination[0], least_expensive_combination[1], marker='o', color=colors[7],
                label='Combi optimum')
        ax.plot(vrm_optimum_point[0], vrm_optimum_point[1], marker='o', color=colors[0], label='VRM optimum',)


    # Shoe line+points for VRM optimization
    ax.plot(vrm_optimization_steps['cost'], vrm_optimization_steps['pf_traject'], color=colors[0],
            label='Optimalisatiestappen', linestyle='-', marker='o', markersize=2)

    # take maximum of the cost and set the x-axis limit
    # ax.set_xlim(left=0, right=df_combinations['cost'].max())
    # ax.set_xlim(left=0, right=4.5e8)
    ax.set_xlim(left=0, right=1e8)
    ax.hlines(p_max, 0, df_combinations['cost'].max(), colors='k', linestyles='dashed', label='Ondergrens')
    ax.set_ylim(top=p_max * 10, bottom=p_max / 10)
    ax.set_xlabel('Kosten (M€)')
    ax.set_ylabel('Traject faalkans in 2075')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    # get xtick labels and divide by 1e6 and replace
    ax.set_xticklabels([f'{x / 1e6:.0f}' for x in ax.get_xticks()])
    ax.grid(True, which='both', linestyle=':')

    # put the legend on the left bottom corner
    ax.legend(loc='lower left')

    plt.show()
    return fig


def plot_sensitivity(df_sensitivity: pd.DataFrame, vrm_optimization_steps: pd.DataFrame, pmax: float):
    fig, ax = plt.subplots()
    from scripts.design.deltares_colors import colors

    ax.scatter(df_sensitivity["cheapest_combination_cost"], df_sensitivity["cheapest_combination_pf"],
               color=colors[4], label='Combi optimum')
    ax.scatter(df_sensitivity["dsn_point_cost"], df_sensitivity["dsn_point_pf"], color=colors[6], label='DSN')
    ax.scatter(df_sensitivity["vrm_ondergrenz_point_cost"], df_sensitivity["vrm_ondergrenz_point_pf"], color=colors[0],
               label='VRM ondergrenz')
    ax.scatter(df_sensitivity["vrm_eco_point_cost"], df_sensitivity["vrm_eco_point_pf"], color=colors[1],
               label='VRM eco opti')

    ax.hlines(pmax, 0, df_sensitivity['dsn_point_cost'].max(), colors='k', linestyles='dashed',
              label='Ondergrens * 0.52')
    ax.plot(vrm_optimization_steps['cost'], vrm_optimization_steps['pf_traject'], color=colors[0],
            label='VR pad basis', linestyle='-', marker='o', markersize=2)

    ax.set_xlim(left=0)
    ax.set_ylim(top=pmax * 10, bottom=pmax / 10)
    ax.set_xlabel('Kosten (M€)')
    ax.set_ylabel('Traject faalkans in 2075')
    ax.set_yscale('log')
    ax.legend()
    plt.show()


def plot_sensitivity_plotly(df_sensitive: pd.DataFrame):
    import plotly.graph_objects as go
    fig = go.Figure()
    from scripts.design.deltares_colors import colors
    fig.add_trace(go.Scatter(x=df_sensitive["least_expensive_combination_cost"],
                             y=df_sensitive["least_expensive_combination_pf"],
                             mode='markers',
                             marker=dict(color=colors[4]),
                             name='Combi optimum'))
    fig.add_trace(go.Scatter(x=df_sensitive["dsn_point_cost"],
                             y=df_sensitive["dsn_point_pf"],
                             mode='markers',
                             marker=dict(color=colors[6]),
                             name='DSN'))
    fig.add_trace(go.Scatter(x=df_sensitive["vrm_eco_point_cost"],
                             y=df_sensitive["vrm_eco_point_pf"],
                             mode='markers',
                             marker=dict(color=colors[0]),
                             name='VRM optimum'))
    fig.update_layout(
        xaxis_title='Kosten (M€)',
        yaxis_title='Traject faalkans in 2075',
        yaxis_type='log',
        showlegend=True
    )
    fig.show()


def plot_histogram_metrics(df_sensitive: pd.DataFrame, normalized: bool, save: bool):
    # Plot one hist per axis, but have multiple axes
    fig, ax = plt.subplots()
    from scripts.design.deltares_colors import colors
    # ax.hist(df_sensitive["dsn_point_cost"] - df_sensitive["vrm_eco_point_cost"], bins=20, color=colors[0], label='dist VR/DSN')
    # remove last element
    if normalized:
        ax.hist((df_sensitive["cheapest_combination_cost"] - df_sensitive["vrm_ondergrenz_point_cost"]) / df_sensitive[
            "vrm_ondergrenz_point_cost"] * 100, bins=20, color=colors[4],
                )
        ax.set_title('Verschil tussen VR (2075) en Optimale uniforme doorsnede-eisen in %')
        ax.set_xlabel('%')

        title = 'Verschil tussen VR en Optimale uniforme doorsnede-eisen genormaliseerd'

    else:
        ax.hist((df_sensitive["cheapest_combination_cost"] - df_sensitive["vrm_ondergrenz_point_cost"]) / 1e6, bins=20,
                color=colors[4],
                )
        ax.set_title('Verschil tussen VR (2075) en Optimale uniforme doorsnede-eisen in M€')
        ax.set_xlabel('Kosten (M€)')
        title = 'Verschil tussen VR en Optimale uniforme doorsnede-eisen'

    ax.set_ylabel('Aantal simulaties')
    ax.legend()
    plt.show()

    if save:
        fig.savefig(title)


def plot_histogram_metrics_respective_to_base(df_sensitive: pd.DataFrame, normalized: bool, save: bool):
    # Plot one hist per axis, but have multiple axes
    fig, ax = plt.subplots()
    from scripts.design.deltares_colors import colors
    # ax.hist(df_sensitive["dsn_point_cost"] - df_sensitive["vrm_eco_point_cost"], bins=20, color=colors[0], label='dist VR/DSN')
    if normalized:
        ax.hist((list(df_sensitive["cheapest_combination_cost"])[0] - df_sensitive["vrm_ondergrenz_point_cost"]) /
                df_sensitive[
                    "vrm_ondergrenz_point_cost"] * 100, bins=20, color=colors[4],
                )
        ax.set_title('Verschil tussen VR (2075) en Optimale uniforme doorsnede-eisen in %')
        title = 'Verschil tussen VR en Optimale uniforme doorsnede-eisen genormaliseerd_compared_to_base'
        ax.set_xlabel('%')


    else:
        ax.hist((list(df_sensitive["cheapest_combination_cost"])[0] - df_sensitive["vrm_ondergrenz_point_cost"]) / 1e6,
                bins=20,
                color=colors[4],
                )
        ax.set_title('Verschil tussen VR (2075) en Optimale uniforme doorsnede-eisen in M€')
        ax.set_xlabel('Kosten (M€)')
        title = 'Verschil tussen VR en Optimale uniforme doorsnede-eisen_compared_to_base'

    ax.set_ylabel('Aantal simulaties')
    ax.legend()
    plt.show()

    if save:
        fig.savefig(title)


def plot_scatter_cost_cost(df_sensitive: pd.DataFrame):
    fig, ax = plt.subplots()
    from scripts.design.deltares_colors import colors
    # ax.scatter(df_sensitive["dsn_point_cost"], df_sensitive["vrm_eco_point_cost"], color=colors[0], label='VRM optimum')
    ax.scatter(df_sensitive["least_expensive_combination_cost"], df_sensitive["dsn_point_cost"], color=colors[4],
               label='data')

    # plot 1:1 line
    ax.plot([2.5e8, 3.5e8], [2.5e8, 3.5e8], color='black', label='1:1 lijn')
    # set axis in Millions
    ax.set_xticklabels([f'{x / 1e6:.0f}' for x in ax.get_xticks()])
    ax.set_yticklabels([f'{x / 1e6:.0f}' for x in ax.get_yticks()])
    ax.set_xlabel('vakspecifiek kosten (M€)')
    ax.set_ylabel('DSN Kosten (M€)')
    ax.legend()
    plt.show()

# df_results = pd.read_csv(Path(__file__).parent.joinpath("old", "results_sensitivity_analysis_38-1.csv"))
# plot_scatter_cost_cost(df_results)
