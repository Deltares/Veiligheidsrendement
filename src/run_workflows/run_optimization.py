import logging
from typing import List

import numpy as np
import pandas as pd

from src.DecisionMaking.Solutions import Solutions
from src.DecisionMaking.Strategy import (
    GreedyStrategy,
    Strategy,
    TargetReliabilityStrategy,
)
from src.run_workflows.vrtool_run_protocol import (
    VrToolPlotMode,
    VrToolRunProtocol,
    VrToolRunResultProtocol,
    load_intermediate_results,
    save_intermediate_results,
)


class RunOptimization(VrToolRunProtocol):
    def __init__(
        self, measures_solutions: List[Solutions], plot_mode: VrToolPlotMode
    ) -> None:
        self._solutions_list = measures_solutions
        self._plot_mode = plot_mode

    def run(self) -> VrToolRunResultProtocol:
        # Either load existing results or compute:
        _final_results_file = self.vr_config.directory.joinpath("FINALRESULT.out.dat")
        if self.vr_config.reuse_output and _final_results_file.exists():
            _results_dict = load_intermediate_results(
                _final_results_file, ["AllStrategies"]
            )
            _all_strategies = _results_dict.pop("AllStrategies")
            logging.info("Loaded AllStrategies from file")
        else:
            ## STEP 3: EVALUATE THE STRATEGIES
            _all_strategies = []
            for i in self.vr_config.design_methods:
                if i in [
                    "TC",
                    "Total Cost",
                    "Optimized",
                    "Greedy",
                    "Veiligheidsrendement",
                ]:
                    # Initialize a GreedyStrategy:
                    _greedy_optimization = GreedyStrategy(i)

                    # Combine available measures
                    _greedy_optimization.combine(
                        self.selected_traject,
                        self._solutions_list,
                        filtering="off",
                        splitparams=True,
                    )

                    # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
                    _greedy_optimization.evaluate(
                        self.selected_traject,
                        self._solutions_list,
                        splitparams=True,
                        setting="cautious",
                        f_cautious=1.5,
                        max_count=600,
                        BCstop=0.1,
                    )

                    # plot beta time for all measure steps for each strategy
                    if self._plot_mode == VrToolPlotMode.EXTENSIVE:
                        _greedy_optimization.plotBetaTime(
                            self.selected_traject,
                            typ="single",
                            path=self.vr_config.directory,
                        )

                    _greedy_optimization = self._replace_names(
                        _greedy_optimization, self._solutions_list
                    )
                    cost_Greedy = _greedy_optimization.determineRiskCostCurve(
                        self.selected_traject
                    )

                    # write to csv's
                    _results_dir = self.vr_config.directory / "results"
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
                            "TakenMeasures_Optimal_"
                            + _greedy_optimization.type
                            + ".csv",
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
                    costs = _greedy_optimization.determineRiskCostCurve(
                        self.selected_traject
                    )
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
                        self.selected_traject,
                        self._solutions_list,
                        filtering="off",
                        splitparams=True,
                    )

                    # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
                    TargetReliabilityBased.evaluate(
                        self.selected_traject, self._solutions_list, splitparams=True
                    )
                    TargetReliabilityBased.makeSolution(
                        _results_dir.joinpath(
                            "FinalMeasures_" + TargetReliabilityBased.type + ".csv",
                        ),
                        type="Final",
                    )

                    # plot beta time for all measure steps for each strategy
                    if self._plot_mode == VrToolPlotMode.EXTENSIVE:
                        TargetReliabilityBased.plotBetaTime(
                            self.selected_traject,
                            typ="single",
                            path=self.vr_config.directory,
                        )

                    TargetReliabilityBased = self._replace_names(
                        TargetReliabilityBased, self._solutions_list
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

        if self.vr_config.shelves:
            save_intermediate_results(
                self.vr_config.directory.joinpath("FINALRESULT.out"),
                {
                    "SelectedTraject": self.selected_traject,
                    "AllSolutions": self._solutions_list,
                    "AllStrategies": _all_strategies,
                },
            )

    def _replace_names(
        self, strategy_case: Strategy, solution_case: Solutions
    ) -> Strategy:
        strategy_case.TakenMeasures = strategy_case.TakenMeasures.reset_index(drop=True)
        for i in range(1, len(strategy_case.TakenMeasures)):
            _measure_id = strategy_case.TakenMeasures.iloc[i]["ID"]
            if isinstance(_measure_id, list):
                _measure_id = "+".join(_measure_id)

            section = strategy_case.TakenMeasures.iloc[i]["Section"]
            name = (
                solution_case[section]
                .MeasureTable.loc[
                    solution_case[section].MeasureTable["ID"] == _measure_id
                ]["Name"]
                .values
            )
            strategy_case.TakenMeasures.at[i, "name"] = name
        return strategy_case
