import time

import matplotlib.pyplot as plt

from src.DecisionMaking.Solutions import Solutions
from src.DecisionMaking.Strategy import GreedyStrategy, TargetReliabilityStrategy

try:
    import cPickle as pickle
except:
    import pickle

import os
import shelve

import numpy as np
import pandas as pd

import src.ProbabilisticTools.ProbabilisticFunctions as ProbabilisticFunctions
from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem.DikeTraject import DikeTraject
from tools.HelperFunctions import replaceNames

"""The function below is the main one for any calculation for SAFE. It contains 3 main steps:
1. Safety assessment
2. Computation of reliability for all solutions per section
3. Optimization of measures in accordance with the chosen design methods

Settings for the computations are imported from config.py and can be set there. 
Note that not all settings are yet generalized, this is work in progress.
"""


def runFullModel(
    TrajectObject: DikeTraject, config: VrtoolConfig, plot_mode: str = "test"
):

    """This is the main routine for a "SAFE"-type calculation
    Input is a TrajectObject = DikeTraject object with all relevant data
    plot_mode sets the amount of plots to be made. 'test' means a simple test approach where only csv's are given as output.
    'standard' means that normal plots are made, and with 'extensive' all plots can be switched on (not recommended)"""

    if config.timing:
        start = time.time()

    # Make a few dirs if they dont exist yet:
    if not config.directory.is_dir():
        config.directory.mkdir(parents=True, exist_ok=True)
        if plot_mode != "test":
            config.directory.joinpath("figures").mkdir(parents=True, exist_ok=True)
        config.directory.joinpath("results", "investment_steps").mkdir(
            parents=True, exist_ok=True
        )

    ## STEP 1: SAFETY ASSESSMENT
    print("Start step 1: safety assessment")

    # Loop over sections and do the assessment.
    for i, section in enumerate(TrajectObject.Sections):
        # get design water level:
        # TODO remove this line?
        # section.Reliability.Load.NormWaterLevel = ProbabilisticFunctions.getDesignWaterLevel(section.Reliability.Load,TrajectObject.GeneralInfo['Pmax'])

        # compute reliability in time for each mechanism:
        # print(section.End)
        for j in TrajectObject.GeneralInfo["MechanismsConsidered"]:
            section.Reliability.Mechanisms[j].generateLCRProfile(
                section.Reliability.Load,
                mechanism=j,
                trajectinfo=TrajectObject.GeneralInfo,
            )

        # aggregate to section reliability:
        section.Reliability.calcSectionReliability()

        # optional: plot reliability in time for each section
        # TODO make a function of this statement to clean code even further.
        if config.plot_reliability_in_time:
            # Plot the initial reliability-time:
            plt.figure(1)
            [
                section.Reliability.Mechanisms[j].drawLCR(mechanism=j)
                for j in config.mechanisms
            ]
            plt.plot(
                [config.t_0, config.t_0 + np.max(config.T)],
                [
                    ProbabilisticFunctions.pf_to_beta(
                        TrajectObject.GeneralInfo["Pmax"]
                    ),
                    ProbabilisticFunctions.pf_to_beta(
                        TrajectObject.GeneralInfo["Pmax"]
                    ),
                ],
                "k--",
                label="Norm",
            )
            plt.legend()
            plt.title(section.name)
            if not config.directory.joinpath("figures", section.name).is_dir():
                config.directory.joinpath("figures", section.name).mkdir(
                    parents=True, exist_ok=True
                )
                config.directory.joinpath("figures", section.name, "Initial")
            plt.savefig(
                config.directory.joinpath(
                    "figures", section.name, "Initial", "InitialSituation" + ".png"
                ),
                bbox_inches="tight",
            )
            plt.close()

    # aggregate computed initial probabilities to DataFrame in TrajectObject:
    TrajectObject.setProbabilities()

    # Plot initial reliability for TrajectObject:
    case_settings = {
        "directory": config.directory,
        "language": config.language,
        "beta_or_prob": config.beta_or_prob,
    }
    if plot_mode != "test":
        TrajectObject.plotAssessment(
            fig_size=(12, 4),
            draw_targetbeta="off",
            last=True,
            t_list=[0, 25, 50],
            case_settings=case_settings,
        )

    print("Finished step 1: assessment of current situation")

    if config.timing:
        print("Time elapsed: " + str(time.time() - start) + " seconds")
        start = time.time()

    # store stuff:
    if config.shelves:
        # Save intermediate results to shelf:
        filename = config.directory.joinpath("AfterStep1.out")
        # make shelf
        my_shelf = shelve.open(str(filename), "n")
        my_shelf["TrajectObject"] = locals()["TrajectObject"]
        my_shelf.close()

    ## STEP 2: INITIALIZE AND EVALUATE MEASURES FOR EACH SECTION
    # Goal: Generate a Measures object with Section name and beta-t-euro relations for each measure. Combining is done later.

    # TODO consider whether combining should be done in this step rather than in the next.
    # Then after selecting a strategy you only need to throw out invalid combinations (e.g., for Target Reliability throw out all investments at t=20

    # Either load existing results or compute:
    if config.reuse_output and os.path.exists(
        config.directory.joinpath("AfterStep2.out.dat")
    ):
        my_shelf = shelve.open(str(config.directory.joinpath("AfterStep2.out")))
        for key in my_shelf:
            AllSolutions = my_shelf[key]
            print("Loaded AllSolutions from file")
        my_shelf.close()
    else:
        AllSolutions = {}
        # Calculate per section, for each measure the cost-reliability-time relations:
        for i in TrajectObject.Sections:
            AllSolutions[i.name] = Solutions(i, config)
            AllSolutions[i.name].fillSolutions(
                config.input_directory.joinpath(i.name + ".xlsx")
            )
            AllSolutions[i.name].evaluateSolutions(i, TrajectObject.GeneralInfo)

    for i in TrajectObject.Sections:
        AllSolutions[i.name].SolutionstoDataFrame(filtering="off", splitparams=True)

    # Store intermediate results:
    if config.shelves:
        filename = config.directory.joinpath("AfterStep2.out")
        # make shelf
        my_shelf = shelve.open(str(filename), "n")
        my_shelf["AllSolutions"] = locals()["AllSolutions"]
        my_shelf.close()

    print("Finished step 2: evaluation of measures")
    if config.timing:
        end = time.time()
        print("Time elapsed: " + str(end - start) + " seconds")
        start = time.time()

    # If desired: plot beta(t)-cost for all measures at a section:
    if config.plot_measure_reliability:
        betaind_array = []

        for i in config.T:
            betaind_array.append("beta" + str(i))

        plt_mech = ["Section", "Piping", "StabilityInner", "Overflow"]

        for i in TrajectObject.Sections:
            for betaind in betaind_array:
                for mech in plt_mech:
                    requiredbeta = ProbabilisticFunctions.pf_to_beta(
                        TrajectObject.GeneralInfo["Pmax"]
                        * (i.Length / TrajectObject.GeneralInfo["TrajectLength"])
                    )
                    plt.figure(1001)
                    AllSolutions[i.name].plotBetaTimeEuro(
                        mechanism=mech,
                        beta_ind=betaind,
                        sectionname=i.name,
                        beta_req=requiredbeta,
                    )
                    plt.savefig(
                        config.directory.joinpath(
                            "figures", i.name, "Measures", mech + "_" + betaind + ".png"
                        ),
                        bbox_inches="tight",
                    )
                    plt.close(1001)
        print("Finished making beta plots")
    # Either load existing results or compute:
    if config.reuse_output and os.path.exists(
        config.directory.joinpath("FINALRESULT.out.dat")
    ):
        my_shelf = shelve.open(str(config.directory.joinpath("FINALRESULT.out")))
        AllStrategies = my_shelf["AllStrategies"]
        print("Loaded AllStrategies from file")

        my_shelf.close()
    else:
        ## STEP 3: EVALUATE THE STRATEGIES
        AllStrategies = []
        for i in config.design_methods:
            if i in ["TC", "Total Cost", "Optimized", "Greedy", "Veiligheidsrendement"]:
                # Initialize a GreedyStrategy:
                GreedyOptimization = GreedyStrategy(i, config)

                # Combine available measures
                GreedyOptimization.combine(
                    TrajectObject, AllSolutions, filtering="off", splitparams=True
                )

                if config.timing:
                    print("Combined measures for " + i)
                    print("Time elapsed: " + str(time.time() - start) + " seconds")
                    start = time.time()

                # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
                GreedyOptimization.evaluate(
                    TrajectObject,
                    AllSolutions,
                    splitparams=True,
                    setting="cautious",
                    f_cautious=1.5,
                    max_count=600,
                    BCstop=0.1,
                )

                # plot beta time for all measure steps for each strategy
                if plot_mode == "extensive":
                    GreedyOptimization.plotBetaTime(
                        TrajectObject, typ="single", path=config.directory
                    )

                GreedyOptimization = replaceNames(GreedyOptimization, AllSolutions)
                cost_Greedy = GreedyOptimization.determineRiskCostCurve(TrajectObject)

                # write to csv's
                GreedyOptimization.TakenMeasures.to_csv(
                    config.directory.joinpath(
                        "results", "TakenMeasures_" + GreedyOptimization.type + ".csv"
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
                    config.directory.joinpath("results", "TotalCostValues_Greedy.csv"),
                    float_format="%.1f",
                )
                GreedyOptimization.makeSolution(
                    config.directory.joinpath(
                        "results",
                        "TakenMeasures_Optimal_" + GreedyOptimization.type + ".csv",
                    ),
                    step=cost_Greedy["TC_min"] + 1,
                    type="Optimal",
                )
                GreedyOptimization.makeSolution(
                    config.directory.joinpath(
                        "results", "FinalMeasures_" + GreedyOptimization.type + ".csv"
                    ),
                    type="Final",
                )
                for j in GreedyOptimization.options:
                    GreedyOptimization.options[j].to_csv(
                        config.directory.joinpath(
                            "results",
                            j + "_Options_" + GreedyOptimization.type + ".csv",
                        )
                    )
                costs = GreedyOptimization.determineRiskCostCurve(TrajectObject)
                TR = costs["TR"]
                LCC = costs["LCC"]
                pd.DataFrame(
                    np.array([TR, LCC]).reshape((len(TR), 2)), columns=["TR", "LCC"]
                ).to_csv(config.directory.joinpath("results", "TotalRiskCost.csv"))
                AllStrategies.append(GreedyOptimization)

                if config.timing:
                    print("Determined strategy for " + i)
                    print("Time elapsed: " + str(time.time() - start) + " seconds")
                    start = time.time()

            elif i in ["OI", "TargetReliability", "Doorsnede-eisen"]:
                # Initialize a strategy type (i.e combination of objective & constraints)
                TargetReliabilityBased = TargetReliabilityStrategy(i)
                # Combine available measures
                TargetReliabilityBased.combine(
                    TrajectObject, AllSolutions, filtering="off", splitparams=True
                )

                if config.timing:
                    print("Combined measures for " + i)
                    print("Time elapsed: " + str(time.time() - start) + " seconds")
                    start = time.time()

                # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
                TargetReliabilityBased.evaluate(
                    TrajectObject, AllSolutions, splitparams=True
                )
                TargetReliabilityBased.makeSolution(
                    config.directory.joinpath(
                        "results",
                        "FinalMeasures_" + TargetReliabilityBased.type + ".csv",
                    ),
                    type="Final",
                )

                # plot beta time for all measure steps for each strategy
                if plot_mode == "extensive":
                    TargetReliabilityBased.plotBetaTime(
                        TrajectObject, typ="single", path=config.directory
                    )

                TargetReliabilityBased = replaceNames(
                    TargetReliabilityBased, AllSolutions
                )
                # write to csv's
                TargetReliabilityBased.TakenMeasures.to_csv(
                    config.directory.joinpath(
                        "results",
                        "TakenMeasures_" + TargetReliabilityBased.type + ".csv",
                    )
                )
                for j in TargetReliabilityBased.options:
                    TargetReliabilityBased.options[j].to_csv(
                        config.directory.joinpath(
                            "results",
                            j + "_Options_" + TargetReliabilityBased.type + ".csv",
                        )
                    )

                AllStrategies.append(TargetReliabilityBased)

                if config.timing:
                    print("Determined strategy for " + i)
                    print("Time elapsed: " + str(time.time() - start) + " seconds")
                    start = time.time()

    if config.shelves:
        # Store final results
        filename = config.directory.joinpath("FINALRESULT.out")

        # make shelf
        my_shelf = shelve.open(str(filename), "n")
        my_shelf["TrajectObject"] = locals()["TrajectObject"]
        my_shelf["AllSolutions"] = locals()["AllSolutions"]
        my_shelf["AllStrategies"] = locals()["AllStrategies"]

        my_shelf.close()
        # TODO as all objects are in this one, the others can possibly be deleted to save space

    return AllStrategies, AllSolutions
