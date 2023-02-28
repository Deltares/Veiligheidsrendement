from __future__ import annotations

import logging
import shelve
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.ProbabilisticTools.ProbabilisticFunctions as pb
from src.DecisionMaking.Solutions import Solutions
from src.DecisionMaking.Strategy import (
    GreedyStrategy,
    Strategy,
    TargetReliabilityStrategy,
)
from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem.DikeSection import DikeSection
from src.FloodDefenceSystem.DikeTraject import DikeTraject

"""
!!IMPORTANT!!
This file is deprecated in favor of the /run_workflows module which should be used instead.
This is just a 'newer' representation of what used to be in /tools/RunModel.py
Use the contents of this file for reference purposes.
"""


def run_model_old_approach(vr_config: VrtoolConfig, plot_mode: str):
    """This is the main routine for a "SAFE"-type calculation
    Input is a TrajectObject = DikeTraject object with all relevant data
    plot_mode sets the amount of plots to be made. 'test' means a simple test approach where only csv's are given as output.
    'standard' means that normal plots are made, and with 'extensive' all plots can be switched on (not recommended)"""
    # Make a few dirs if they dont exist yet:
    if not vr_config.directory.is_dir():
        vr_config.directory.mkdir(parents=True, exist_ok=True)
        if plot_mode != "test":
            vr_config.directory.joinpath("figures").mkdir(parents=True, exist_ok=True)
        vr_config.directory.joinpath("results", "investment_steps").mkdir(
            parents=True, exist_ok=True
        )

    _selected_traject = _load_traject(vr_config)
    _step_safety_assessment(vr_config, _selected_traject, plot_mode, vr_config.shelves)
    _measures_solutions = _step_measures(
        vr_config, _selected_traject, vr_config.shelves
    )
    _strategies = _step_optimization(
        vr_config, _selected_traject, _measures_solutions, plot_mode, vr_config.shelves
    )

    return _strategies, _measures_solutions


def _load_traject(vr_config: VrtoolConfig) -> DikeTraject:
    _traject = DikeTraject(traject=vr_config.traject)
    _traject.ReadAllTrajectInput(vr_config.path)
    return _traject


def _save_intermediate_results(filename: Path, results_dict: dict) -> None:
    # make shelf
    my_shelf = shelve.open(str(filename), "n")
    for key, value in results_dict.items():
        my_shelf[key] = value
    my_shelf.close()


def _load_intermediate_results(filename: Path, results_keys: List[str]) -> dict:
    _shelf = shelve.open(str(filename))
    _result_dict = {
        _result_key: _shelf[_result_key]
        for _result_key in results_keys
        if _result_key in _shelf.keys()
    }
    _shelf.close()
    return _result_dict


def _step_safety_assessment(
    vr_config: VrtoolConfig,
    selected_traject: DikeTraject,
    plot_mode: str,
    save_to_file: bool,
) -> None:
    ## STEP 1: SAFETY ASSESSMENT
    logging.info("Start step 1: safety assessment")

    # Loop over sections and do the assessment.
    for _, section in enumerate(selected_traject.Sections):
        # get design water level:
        # TODO remove this line?
        # section.Reliability.Load.NormWaterLevel = pb.getDesignWaterLevel(section.Reliability.Load,selected_traject.GeneralInfo['Pmax'])

        # compute reliability in time for each mechanism:
        # logging.info(section.End)
        for j in selected_traject.GeneralInfo["MechanismsConsidered"]:
            section.Reliability.Mechanisms[j].generateLCRProfile(
                section.Reliability.Load,
                mechanism=j,
                trajectinfo=selected_traject.GeneralInfo,
            )

        # aggregate to section reliability:
        section.Reliability.calcSectionReliability()

        # optional: plot reliability in time for each section
        if vr_config.plot_reliability_in_time:
            _plot_reliability_in_time(section, selected_traject, vr_config)

    # aggregate computed initial probabilities to DataFrame in selected_traject:
    selected_traject.setProbabilities()

    # Plot initial reliability for selected_traject:
    case_settings = {
        "directory": vr_config.directory,
        "language": vr_config.language,
        "beta_or_prob": vr_config.beta_or_prob,
    }
    if plot_mode != "test":
        selected_traject.plotAssessment(
            fig_size=(12, 4),
            draw_targetbeta="off",
            last=True,
            t_list=[0, 25, 50],
            case_settings=case_settings,
        )

    logging.info("Finished step 1: assessment of current situation")

    # store stuff:
    if save_to_file:
        # Save intermediate results to shelf:
        _save_intermediate_results(
            vr_config.directory.joinpath("AfterStep1.out"),
            dict(SelectedTraject=selected_traject),
        )


def _step_measures(
    vr_config: VrtoolConfig, selected_traject: DikeTraject, save_to_file: bool
) -> Dict[str, Solutions]:
    ## STEP 2: INITIALIZE AND EVALUATE MEASURES FOR EACH SECTION
    # Goal: Generate a Measures object with Section name and beta-t-euro relations for each measure. Combining is done later.

    # TODO consider whether combining should be done in this step rather than in the next.
    # Then after selecting a strategy you only need to throw out invalid combinations (e.g., for Target Reliability throw out all investments at t=20

    # Either load existing results or compute:
    _step_two_output = vr_config.directory.joinpath("AfterStep2.out.dat")
    if vr_config.reuse_output and _step_two_output.exists():
        _results_dict = _load_intermediate_results(_step_two_output, ["AllSolutions"])
        logging.info("Loaded AllSolutions from file")
        _all_solutions = _results_dict.pop["AllSolutions"]
    else:
        _all_solutions = {}
        # Calculate per section, for each measure the cost-reliability-time relations:
        for i in selected_traject.Sections:
            _all_solutions[i.name] = Solutions(i)
            _all_solutions[i.name].fillSolutions(
                vr_config.path.joinpath(i.name + ".xlsx")
            )
            _all_solutions[i.name].evaluateSolutions(i, selected_traject.GeneralInfo)

    for i in selected_traject.Sections:
        _all_solutions[i.name].SolutionstoDataFrame(filtering="off", splitparams=True)

    # Store intermediate results:
    if save_to_file:
        filename = vr_config.directory.joinpath("AfterStep2.out")
        _save_intermediate_results(filename, dict(AllSolutions=_all_solutions))

    logging.info("Finished step 2: evaluation of measures")

    # If desired: plot beta(t)-cost for all measures at a section:
    if vr_config.plot_measure_reliability:
        _plot_measure_reliability(vr_config, selected_traject, _all_solutions)
    return _all_solutions


def _step_optimization(
    vr_config: VrtoolConfig,
    selected_traject: DikeTraject,
    solutions_list: List[Solutions],
    plot_mode: str,
    save_to_file: bool,
) -> List[Strategy]:
    # Either load existing results or compute:
    _final_results_file = vr_config.directory.joinpath("FINALRESULT.out.dat")
    if vr_config.reuse_output and _final_results_file.exists():
        _results_dict = _load_intermediate_results(
            _final_results_file, ["AllStrategies"]
        )
        _all_strategies = _results_dict.pop("AllStrategies")
        logging.info("Loaded AllStrategies from file")
    else:
        ## STEP 3: EVALUATE THE STRATEGIES
        _all_strategies = []
        for i in vr_config.design_methods:
            if i in ["TC", "Total Cost", "Optimized", "Greedy", "Veiligheidsrendement"]:
                # Initialize a GreedyStrategy:
                _greedy_optimization = GreedyStrategy(i)

                # Combine available measures
                _greedy_optimization.combine(
                    selected_traject, solutions_list, filtering="off", splitparams=True
                )

                # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
                _greedy_optimization.evaluate(
                    selected_traject,
                    solutions_list,
                    splitparams=True,
                    setting="cautious",
                    f_cautious=1.5,
                    max_count=600,
                    BCstop=0.1,
                )

                # plot beta time for all measure steps for each strategy
                if plot_mode == "extensive":
                    _greedy_optimization.plotBetaTime(
                        selected_traject, typ="single", path=vr_config.directory
                    )

                _greedy_optimization = _replace_names(
                    _greedy_optimization, solutions_list
                )
                cost_Greedy = _greedy_optimization.determineRiskCostCurve(
                    selected_traject
                )

                # write to csv's
                _results_dir = vr_config.directory / "results"
                _greedy_optimization.TakenMeasures.to_csv(
                    _results_dir.joinpath(
                        "TakenMeasures_" + _greedy_optimization.type + ".csv"
                    )
                )
                pd.DataFrame(
                    np.array(
                        [
                            cost_Greedy["LCC"],
                            cost_Greedy["TR"],
                            np.add(cost_Greedy["LCC"], cost_Greedy["TR"]),
                        ]
                    ).T,
                    columns=["LCC", "TR", "TC"],
                ).to_csv(
                    _results_dir / "TotalCostValues_Greedy.csv",
                    float_format="%.1f",
                )
                _greedy_optimization.makeSolution(
                    _results_dir.joinpath(
                        "TakenMeasures_Optimal_" + _greedy_optimization.type + ".csv",
                    ),
                    step=cost_Greedy["TC_min"] + 1,
                    type="Optimal",
                )
                _greedy_optimization.makeSolution(
                    _results_dir.joinpath(
                        "FinalMeasures_" + _greedy_optimization.type + ".csv"
                    ),
                    type="Final",
                )
                for j in _greedy_optimization.options:
                    _greedy_optimization.options[j].to_csv(
                        _results_dir.joinpath(
                            j + "_Options_" + _greedy_optimization.type + ".csv",
                        )
                    )
                costs = _greedy_optimization.determineRiskCostCurve(selected_traject)
                TR = costs["TR"]
                LCC = costs["LCC"]
                pd.DataFrame(
                    np.array([TR, LCC]).reshape((len(TR), 2)), columns=["TR", "LCC"]
                ).to_csv(_results_dir / "TotalRiskCost.csv")
                _all_strategies.append(_greedy_optimization)

            elif i in ["OI", "TargetReliability", "Doorsnede-eisen"]:
                # Initialize a strategy type (i.e combination of objective & constraints)
                TargetReliabilityBased = TargetReliabilityStrategy(i)
                # Combine available measures
                TargetReliabilityBased.combine(
                    selected_traject, solutions_list, filtering="off", splitparams=True
                )

                # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
                TargetReliabilityBased.evaluate(
                    selected_traject, solutions_list, splitparams=True
                )
                TargetReliabilityBased.makeSolution(
                    _results_dir.joinpath(
                        "FinalMeasures_" + TargetReliabilityBased.type + ".csv",
                    ),
                    type="Final",
                )

                # plot beta time for all measure steps for each strategy
                if plot_mode == "extensive":
                    TargetReliabilityBased.plotBetaTime(
                        selected_traject, typ="single", path=vr_config.directory
                    )

                TargetReliabilityBased = _replace_names(
                    TargetReliabilityBased, solutions_list
                )
                # write to csv's
                TargetReliabilityBased.TakenMeasures.to_csv(
                    _results_dir.joinpath(
                        "TakenMeasures_" + TargetReliabilityBased.type + ".csv",
                    )
                )
                for j in TargetReliabilityBased.options:
                    TargetReliabilityBased.options[j].to_csv(
                        _results_dir.joinpath(
                            j + "_Options_" + TargetReliabilityBased.type + ".csv",
                        )
                    )

                _all_strategies.append(TargetReliabilityBased)

    if save_to_file:
        _save_intermediate_results(
            vr_config.directory.joinpath("FINALRESULT.out"),
            {
                "SelectedTraject": selected_traject,
                "AllSolutions": solutions_list,
                "AllStrategies": _all_strategies,
            },
        )


def _plot_measure_reliability(
    vr_config: VrtoolConfig, selected_traject: DikeTraject, measure_solutions: list
):
    """
    Plot related to step 2
    Args:
        vr_config (VrtoolConfig): _description_
        selected_traject (DikeTraject): _description_
        measure_solutions (list): _description_
    """
    betaind_array = []

    for i in vr_config.T:
        betaind_array.append("beta" + str(i))

    plt_mech = ["Section", "Piping", "StabilityInner", "Overflow"]

    for i in selected_traject.Sections:
        for betaind in betaind_array:
            for mech in plt_mech:
                requiredbeta = pb.pf_to_beta(
                    selected_traject.GeneralInfo["Pmax"]
                    * (i.Length / selected_traject.GeneralInfo["TrajectLength"])
                )
                plt.figure(1001)
                measure_solutions[i.name].plotBetaTimeEuro(
                    mechanism=mech,
                    beta_ind=betaind,
                    sectionname=i.name,
                    beta_req=requiredbeta,
                )
                plt.savefig(
                    vr_config.directory.joinpath(
                        "figures", i.name, "Measures", mech + "_" + betaind + ".png"
                    ),
                    bbox_inches="tight",
                )
                plt.close(1001)
    logging.info("Finished making beta plots")


def _plot_reliability_in_time(
    section: DikeSection, selected_traject: DikeTraject, vr_config: VrtoolConfig
):
    """
    Plotting related to step 1.
    Args:
        section (DikeSection): _description_
        selected_traject (DikeTraject): _description_
        vr_config (VrtoolConfig): _description_
    """
    # if vr_config.plot_reliability_in_time:
    # Plot the initial reliability-time:
    plt.figure(1)
    [
        section.Reliability.Mechanisms[j].drawLCR(mechanism=j)
        for j in vr_config.mechanisms
    ]
    plt.plot(
        [vr_config.t_0, vr_config.t_0 + np.max(vr_config.T)],
        [
            pb.pf_to_beta(selected_traject.GeneralInfo["Pmax"]),
            pb.pf_to_beta(selected_traject.GeneralInfo["Pmax"]),
        ],
        "k--",
        label="Norm",
    )
    plt.legend()
    plt.title(section.name)
    _section_figures_dir = vr_config.directory / "figures" / section.name
    if not _section_figures_dir.exists():
        _section_figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        _section_figures_dir / "Initial" / "InitialSituation.png",
        bbox_inches="tight",
    )
    plt.close()


def _replace_names(strategy_case, solution_case: Solutions):
    strategy_case.TakenMeasures = strategy_case.TakenMeasures.reset_index(drop=True)
    for i in range(1, len(strategy_case.TakenMeasures)):
        _measure_id = strategy_case.TakenMeasures.iloc[i]["ID"]
        if isinstance(_measure_id, list):
            _measure_id = "+".join(_measure_id)

        section = strategy_case.TakenMeasures.iloc[i]["Section"]
        name = (
            solution_case[section]
            .MeasureTable.loc[solution_case[section].MeasureTable["ID"] == _measure_id][
                "Name"
            ]
            .values
        )
        strategy_case.TakenMeasures.at[i, "name"] = name
    return strategy_case
