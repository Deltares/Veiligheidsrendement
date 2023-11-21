import logging
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies import GreedyStrategy, TargetReliabilityStrategy
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)
from vrtool.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class RunOptimization(VrToolRunProtocol):
    def __init__(
        self,
        results_measures: ResultsMeasures,
        optimization_selected_measure_ids: dict[int, list[int]],
    ) -> None:
        if not isinstance(results_measures, ResultsMeasures):
            raise ValueError(
                "Required valid instance of {} as an argument.".format(
                    ResultsMeasures.__name__
                )
            )

        self.selected_traject = results_measures.selected_traject
        self.vr_config = results_measures.vr_config
        self.run_ids = list(optimization_selected_measure_ids.keys())
        self._selected_measure_ids = optimization_selected_measure_ids
        self._solutions_dict = results_measures.solutions_dict
        self._ids_to_import = results_measures.ids_to_import

    def _get_output_dir(self) -> Path:
        _results_dir = self.vr_config.output_directory
        if not _results_dir.exists():
            _results_dir.mkdir(parents=True)
        return _results_dir

    def _get_optimized_greedy_strategy(self, design_method: str) -> StrategyBase:
        # Initialize a GreedyStrategy:
        _greedy_optimization = GreedyStrategy(
            design_method, self.vr_config
            )
        
        _results_dir = self._get_output_dir()
        _greedy_optimization.set_investment_years(
            self.selected_traject,
            self._ids_to_import,
            self._selected_measure_ids,
            self._solutions_dict,
        )
        # Combine available measures
        _greedy_optimization.combine(
            self.selected_traject,
            self._solutions_dict,
            filtering="off",
            splitparams=True,
        )

        # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
        _greedy_optimization.evaluate(
            self.selected_traject,
            self._solutions_dict,
            splitparams=True,
            setting="cautious",
            f_cautious=1.5,
            max_count=600,
            BCstop=0.1,
        )

        _greedy_optimization = self._replace_names(
            _greedy_optimization, self._solutions_dict
        )
        _cost_greedy = _greedy_optimization.determine_risk_cost_curve(
            self.selected_traject.general_info.FloodDamage, None
        )

        _greedy_optimization.write_reliability_to_csv(_results_dir, "Greedy")
        # write to csv's
        _greedy_optimization.TakenMeasures.to_csv(
            _results_dir.joinpath("TakenMeasures_" + _greedy_optimization.type + ".csv")
        )
        pd.DataFrame(
            np.array(
                [
                    _cost_greedy["LCC"],
                    _cost_greedy["TR"],
                    np.add(_cost_greedy["LCC"], _cost_greedy["TR"]),
                ]
            ).T,
            columns=["LCC", "TR", "TC"],
        ).to_csv(
            _results_dir / "TotalCostValues_Greedy.csv",
            float_format="%.1f",
        )
        _greedy_optimization.make_solution(
            _results_dir.joinpath(
                "TakenMeasures_Optimal_" + _greedy_optimization.type + ".csv",
            ),
            step=_cost_greedy["TC_min"] + 1,
            type="Optimal",
        )
        _greedy_optimization.make_solution(
            _results_dir.joinpath(
                "FinalMeasures_" + _greedy_optimization.type + ".csv"
            ),
            type="Final",
        )
        for j in _greedy_optimization.options:
            _greedy_optimization.options[j].to_csv(
                _results_dir.joinpath(
                    j + "_Options_" + _greedy_optimization.type + ".csv",
                ),
                float_format="%.3f",
            )

        return _greedy_optimization

    def _get_target_reliability_strategy(self, design_method: str) -> StrategyBase:
        # Initialize a strategy type (i.e combination of objective & constraints)
        _target_reliability_based = TargetReliabilityStrategy(
            design_method, self.vr_config
        )
        _results_dir = self._get_output_dir()
        
        #filter those measures that are not available at the first available time step
        self._filter_measures_first_time()
        
        _target_reliability_based.set_investment_years(
            self.selected_traject,
            self._ids_to_import,
            self._selected_measure_ids,
            self._solutions_dict,
        )

        # Combine available measures
        _target_reliability_based.combine(
            self.selected_traject,
            self._solutions_dict,
            filtering="off",
            splitparams=True,
        )

        # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
        _target_reliability_based.evaluate(
            self.selected_traject, self._solutions_dict, splitparams=True
        )
        _target_reliability_based.make_solution(
            _results_dir.joinpath(
                "FinalMeasures_" + _target_reliability_based.type + ".csv",
            ),
            type="Final",
        )

        _target_reliability_based = self._replace_names(
            _target_reliability_based, self._solutions_dict
        )
        # write to csv's
        _target_reliability_based.TakenMeasures.to_csv(
            _results_dir.joinpath(
                "TakenMeasures_" + _target_reliability_based.type + ".csv",
            )
        )
        for j in _target_reliability_based.options:
            _target_reliability_based.options[j].to_csv(
                _results_dir.joinpath(
                    j + "_Options_" + _target_reliability_based.type + ".csv",
                ),
                float_format="%.3f",
            )

        return _target_reliability_based

    def _get_evaluation_mapping(self) -> Dict[str, Callable[[str], StrategyBase]]:
        return {
            "TC": self._get_optimized_greedy_strategy,
            "Total Cost": self._get_optimized_greedy_strategy,
            "Optimized": self._get_optimized_greedy_strategy,
            "Greedy": self._get_optimized_greedy_strategy,
            "Veiligheidsrendement": self._get_optimized_greedy_strategy,
            "OI": self._get_target_reliability_strategy,
            "TargetReliability": self._get_target_reliability_strategy,
            "Doorsnede-eisen": self._get_target_reliability_strategy,
        }

    def run(self) -> ResultsOptimization:
        logging.info("Start step 3: Optimization")
        _results_optimization = ResultsOptimization()
        _results_optimization.vr_config = self.vr_config

        ## STEP 3: EVALUATE THE STRATEGIES
        _evaluation_mapping = self._get_evaluation_mapping()
        _results_optimization.results_strategies.extend(
            [
                _evaluation_mapping[_design_method](_design_method)
                for _design_method in self.vr_config.design_methods
                if _design_method in _evaluation_mapping.keys()
            ]
        )

        logging.info("Finished step 3: Optimization")
        _results_optimization.selected_traject = self.selected_traject
        _results_optimization.results_solutions = self._solutions_dict

        return _results_optimization

    def _replace_names(
        self, strategy_case: StrategyBase, solution_case: Solutions
    ) -> StrategyBase:
        strategy_case.TakenMeasures = strategy_case.TakenMeasures.reset_index(drop=True)
        for i in range(1, len(strategy_case.TakenMeasures)):
            _measure_id = strategy_case.TakenMeasures.iloc[i]["ID"]
            if isinstance(_measure_id, list):
                _measure_id = "+".join(_measure_id)

            section = strategy_case.TakenMeasures.iloc[i]["Section"]
            name = (
                solution_case[section]
                .measure_table.loc[
                    solution_case[section].measure_table["ID"] == _measure_id
                ]["Name"]
                .values
            )
            strategy_case.TakenMeasures.at[i, "name"] = name
        return strategy_case

    def _filter_measures_first_time(self):
        '''Filter measures that are not in the first time step that is available for the measure as these should not be included for target reliability strategy'''
        min_dict = {} #dict to store measure for ids_to_import
        count_dict = {} #dict to store counter for selected_measure_ids
        run_id = list(self._selected_measure_ids.keys())[0]
        for counter, (id, value) in enumerate(self._ids_to_import):
            if (id not in min_dict) or (value < min_dict[id]):
                min_dict[id] = value
                count_dict[id] = counter

        self._ids_to_import = [(id, value) for id, value in min_dict.items()]
        self._selected_measure_ids[run_id] = [self._selected_measure_ids[run_id][index] for index in count_dict.values()]

        #filter solutions_dict
        for section in self._solutions_dict.keys():
            _min_year = min(self._solutions_dict[section].MeasureData["year"])
            self._solutions_dict[section].MeasureData = self._solutions_dict[section].MeasureData.loc[self._solutions_dict[section].MeasureData["year"] == _min_year].reset_index(drop=True)

