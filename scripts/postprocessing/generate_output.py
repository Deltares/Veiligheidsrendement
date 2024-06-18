import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import axis

from vrtool.common.enums import MechanismEnum


def plot_lcc_tc_from_steps(
    steps_dict: list[dict], axis: axis, lbl: str, clr: str, mrkr: str = "o"
):
    """Plot the total_lcc and total_risk from the optimization steps. It will give a

    Args:
    steps_dict: list of dicts, each dict contains the optimization step number, data on total_lcc and total_risk
    axis: matplotlib axis object
    lbl: str, label for the plot
    clr: str, color for the plot
    mrkr: str, marker for the plot

    Returns:
    None

    """
    for count, step in enumerate(steps_dict):
        if count == 0:
            axis.plot(
                step["total_lcc"],
                step["total_risk"],
                label=lbl,
                color=clr,
                marker=mrkr,
                markersize=0.5,
            )
        else:
            axis.plot(
                step["total_lcc"],
                step["total_risk"],
                color=clr,
                marker=mrkr,
                markersize=0.5,
            )


def plot_traject_probability_for_step(
    traject_prob_step, ax, run_label="", color="k", linestyle="--"
):
    """Plot the probability of failure for each mechanism for each time step.

    Args:
    traject_prob_step: dict, dictionary containing the probability of failure for each mechanism at each time step
    ax: matplotlib axis object
    run_label: str, label for the line to be plotted
    color: str, color for the plotted line
    linestyle: str, linestyle for the plotted line

    Returns:
    None
    """

    def calculate_traject_probability(traject_prob):
        p_nonf = [1] * len(list(traject_prob.values())[0].values())
        for _mechanism, _data in traject_prob.items():
            if not _data:
                # Sometimes revetment is included yet with no data.
                print(f"No information related to mechanism {_mechanism}")
                continue
            time, pf = zip(*sorted(_data.items()))
            p_nonf = np.multiply(p_nonf, np.subtract(1, pf))
        return time, list(1 - p_nonf)

    time, pf_traject = calculate_traject_probability(traject_prob_step)
    ax.plot(time, pf_traject, label=run_label, color=color, linestyle=linestyle)

    # for mechanism, data in traject_prob_step.items():
    #     time, pf = zip(*sorted(data.items()))
    #     ax.plot(time, pf, label = f'{mechanism.name.capitalize()} {run_label}')
    ax.set_yscale("log")
    ax.set_xlabel("Tijd")
    ax.set_ylabel("Faalkans")
    ax.legend()


def measure_per_section_to_df(measures_per_section, section_parameters):
    """Convert the measures per section to a pandas dataframe.

    Args:
    measures_per_section: dict, dictionary containing the measures per section.
    section_parameters: dict, dictionary containing the parameters per section.

    Returns:
    pd.DataFrame, dataframe containing the measures per section.

    """

    def get_LCC(parameters, investment_years, r=1.03):
        LCC = 0
        for count, parameterset in enumerate(parameters):
            LCC += parameterset["cost"] / (r ** investment_years[count])
        return LCC

    def concatenate_names(parameters):
        return " + ".join([parameter["name"] for parameter in parameters])

    def concatenate_investment_years(investment_years):
        return " + ".join(
            [str(investment_year) for investment_year in investment_years]
        )

    def get_parameters(parameters):
        # for each measure in parameters, if the key is there, add it to the combined dict
        parameter_dict = {}
        for parameter in parameters:
            parameter_dict.update(parameter)
        del parameter_dict["name"]
        del parameter_dict["cost"]
        return parameter_dict

    df = pd.DataFrame(
        columns=[
            "section_id",
            "name",
            "LCC",
            "dcrest",
            "dberm",
            "beta_target",
            "transition_level",
        ]
    )
    for section in section_parameters.keys():
        _LCC = get_LCC(section_parameters[section], measures_per_section[section][1])
        _name = concatenate_names(section_parameters[section])
        _investment_years = concatenate_investment_years(
            measures_per_section[section][1]
        )
        _parameters = get_parameters(section_parameters[section])
        # append to df using concat
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "section_id": section,
                        "name": _name,
                        "LCC": _LCC,
                        "investment_years": _investment_years,
                        **_parameters,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

    return df


def plot_comparison_of_beta_values(betas_per_section_and_mechanism):
    """Plot the beta values for the different mechanisms per section.

    Args:
    betas_per_section_and_mechanism: pd.DataFrame, dataframe containing the beta values for the different mechanisms per section.
    Should contain columns: section_id, mechanism, beta, run

    Returns:
    None
    """
    colors = sns.color_palette("colorblind", 8)
    fig, ax = plt.subplots(nrows=4, figsize=(10, 8))
    for count, mechanism in enumerate(
        betas_per_section_and_mechanism["mechanism"].unique()
    ):
        sns.barplot(
            data=betas_per_section_and_mechanism[
                (betas_per_section_and_mechanism["mechanism"] == mechanism)
            ],
            x="section_id",
            y="beta",
            hue="run",
            ax=ax[count],
            palette=[colors[2], colors[3]],
        )
        ax[count].set_title(mechanism, size="small")
        ax[count].set_ylabel("Beta", size="x-small")
        ax[count].set_xlabel("")
        ax[count].set_ylim(bottom=3, top=6)
        # remove legend
        ax[count].get_legend().remove()
    ax[count].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=2, fontsize="x-small"
    )
    ax[count].set_xlabel("Section ID", size="x-small")
    plt.subplots_adjust(hspace=0.5)


def plot_difference_in_betas(
    betas_per_section_and_mechanism, has_revetment, run_1="reference", run_2="result"
):
    """Plot the difference in beta values between two runs.

    Args:
    betas_per_section_and_mechanism: pd.DataFrame, dataframe containing the beta values for the different mechanisms per section.
    Should contain columns: section_id, mechanism, beta, run
    has_revetment: bool, whether the traject has a revetment
    run_1: str, name of the first run
    run_2: str, name of the second run

    Returns:
    None
    """
    fig, ax = plt.subplots()
    # diagonal line
    ax.plot(
        [min(betas_per_section_and_mechanism.beta), 8],
        [min(betas_per_section_and_mechanism.beta), 8],
        "k--",
    )
    for mechanism in [
        MechanismEnum.OVERFLOW,
        MechanismEnum.PIPING,
        MechanismEnum.STABILITY_INNER,
        MechanismEnum.REVETMENT,
    ]:
        if has_revetment or mechanism != MechanismEnum.REVETMENT:
            ax.plot(
                betas_per_section_and_mechanism.loc[
                    (betas_per_section_and_mechanism["mechanism"] == mechanism)
                    & (betas_per_section_and_mechanism["run"] == run_1)
                ].beta,
                betas_per_section_and_mechanism.loc[
                    (betas_per_section_and_mechanism["mechanism"] == mechanism)
                    & (betas_per_section_and_mechanism["run"] == run_2)
                ].beta,
                "o",
                label=mechanism,
            )
    ax.legend()
    ax.set_xlabel(f"Beta {run_1}")
    ax.set_ylabel(f"Beta {run_2}")


def plot_difference_in_betas_per_section(
    betas_per_section_and_mechanism, has_revetment, run_1="reference", run_2="result"
):
    """Plot the difference in beta values between two runs per section.

    Args:
    betas_per_section_and_mechanism: pd.DataFrame, dataframe containing the beta values for the different mechanisms per section.
    Should contain columns: section_id, mechanism, beta, run
    has_revetment: bool, whether the traject has a revetment
    run_1: str, name of the first run
    run_2: str, name of the second run

    Returns:
    None
    """
    fig, ax = plt.subplots()
    # markers
    markers = ["d", "o", "s", "v"]
    for count, mechanism in enumerate(
        [
            MechanismEnum.OVERFLOW,
            MechanismEnum.PIPING,
            MechanismEnum.STABILITY_INNER,
            MechanismEnum.REVETMENT,
        ]
    ):
        if has_revetment or mechanism != MechanismEnum.REVETMENT:
            ax.plot(
                betas_per_section_and_mechanism.loc[
                    (betas_per_section_and_mechanism["mechanism"] == mechanism)
                    & (betas_per_section_and_mechanism["run"] == run_1)
                ].section_id,
                betas_per_section_and_mechanism.loc[
                    (betas_per_section_and_mechanism["mechanism"] == mechanism)
                    & (betas_per_section_and_mechanism["run"] == run_2)
                ].beta.values
                - betas_per_section_and_mechanism.loc[
                    (betas_per_section_and_mechanism["mechanism"] == mechanism)
                    & (betas_per_section_and_mechanism["run"] == run_1)
                ].beta.values,
                markers[count],
                label=mechanism,
            )
    ax.legend()
    ax.set_xlabel("Section ID")
    ax.set_ylabel("Beta difference")
    ax.set_title(
        f"Difference of beta between {run_1} and {run_2} (positive means {run_2} is higher)"
    )
