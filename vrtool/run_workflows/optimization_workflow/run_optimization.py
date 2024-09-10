import logging
from pathlib import Path
from typing import Callable

from vrtool.decision_making.strategies import (
    GreedyStrategy,
    TargetReliabilityStrategy,
    SmartTargetReliabilityStrategy,
)
from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.optimization.controllers.strategy_controller import StrategyController
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.run_workflows.optimization_workflow.optimization_input_measures import (
    OptimizationInputMeasures,
)
from vrtool.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)
from vrtool.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class RunOptimization(VrToolRunProtocol):
    _strategy_controller: StrategyController
    _selected_measure_ids: dict[int, list[int]]
    _ids_to_import: list[tuple[int, int]]

    def __init__(
        self,
        optimization_input: OptimizationInputMeasures,
        optimization_selected_measure_ids: dict[int, list[int]],
    ) -> None:
        if not isinstance(optimization_input, OptimizationInputMeasures):
            raise ValueError(
                "Required valid instance of {} as an argument.".format(
                    OptimizationInputMeasures.__name__
                )
            )

        self.selected_traject = optimization_input.selected_traject
        self.vr_config = optimization_input.vr_config
        self.vr_config.validate_config()
        self._strategy_controller = self._get_strategy_controller_with_aggregations(
            optimization_input.section_input_collection
        )
        self._selected_measure_ids = optimization_selected_measure_ids
        self._ids_to_import = optimization_input.measure_id_year_list

    def _get_output_dir(self) -> Path:
        _results_dir = self.vr_config.output_directory
        if not _results_dir.exists():
            _results_dir.mkdir(parents=True)
        return _results_dir

    def _get_strategy_controller_with_aggregations(
        self,
        section_input_collection: list[SectionAsInput],
    ) -> StrategyController:
        _strategy_controller = StrategyController(
            section_input_collection, self.vr_config
        )
        _strategy_controller.set_investment_year()
        _strategy_controller.combine()
        _strategy_controller.aggregate()
        return _strategy_controller

    def _get_optimized_greedy_strategy(self, design_method: str) -> StrategyProtocol:
        logging.info("Start optimalisatie van maatregelen voor %s.", design_method)

        # Initalize strategy controller
        _greedy_optimization_input = self._strategy_controller.get_evaluate_input(
            GreedyStrategy, design_method
        )

        # Initialize a GreedyStrategy:
        _greedy_strategy = GreedyStrategy(_greedy_optimization_input, self.vr_config)

        _greedy_strategy.evaluate(
            setting="cautious",
            f_cautious=1.5,
            max_count=600,
            BCstop=0.1,
        )
        return _greedy_strategy

    def _get_smart_target_reliability_strategy(
        self, design_method: str
    ) -> StrategyProtocol:
        logging.info("Start bepaling maatregelen op basis van %s.", design_method)
        # Initalize strategy controller
        _smart_target_reliability_input = self._strategy_controller.get_evaluate_input(
            SmartTargetReliabilityStrategy, design_method
        )

        # Initialize a strategy type (i.e combination of objective & constraints)
        _smart_target_reliability_based = SmartTargetReliabilityStrategy(
            _smart_target_reliability_input, self.vr_config
        )

        # filter those measures that are not available at the first available time step
        # self._filter_measures_first_time()

        # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
        _smart_target_reliability_based.evaluate(self.selected_traject)
        return _smart_target_reliability_based

    def _get_target_reliability_strategy(self, design_method: str) -> StrategyProtocol:
        logging.info(
            "Start bepaling referentiemaatregelen op basis van %s.", design_method
        )
        # Initalize strategy controller
        _target_reliability_input = self._strategy_controller.get_evaluate_input(
            TargetReliabilityStrategy, design_method
        )

        # Initialize a strategy type (i.e combination of objective & constraints)
        _target_reliability_based = TargetReliabilityStrategy(
            _target_reliability_input, self.vr_config
        )

        # filter those measures that are not available at the first available time step
        # self._filter_measures_first_time()

        # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
        _target_reliability_based.evaluate(self.selected_traject)
        return _target_reliability_based

    def _get_evaluation_mapping(self) -> dict[str, Callable[[str], StrategyProtocol]]:
        return {
            "TC": self._get_optimized_greedy_strategy,
            "Total Cost": self._get_optimized_greedy_strategy,
            "Optimized": self._get_optimized_greedy_strategy,
            "Greedy": self._get_optimized_greedy_strategy,
            "Veiligheidsrendement": self._get_optimized_greedy_strategy,
            "OI": self._get_target_reliability_strategy,
            "TargetReliability": self._get_target_reliability_strategy,
            "Doorsnede-eisen": self._get_target_reliability_strategy,
            "Specifieke doorsnede-eisen": self._get_smart_target_reliability_strategy,
        }

    def run(self) -> ResultsOptimization:
        logging.info("Start stap 3: Bepaling maatregelen op trajectniveau.")
        _results_optimization = ResultsOptimization()
        _results_optimization.vr_config = self.vr_config
        # TODO (VRTOOL-406): Selected traject is not required for exporting a result optimization.
        # it is, however, required by the `VrToolRunResultProtocol` implemented by
        # `ResultsOptimization`.
        _results_optimization.selected_traject = self.selected_traject

        ## STEP 3: EVALUATE THE STRATEGIES
        _evaluation_mapping = self._get_evaluation_mapping()
        _results_optimization.results_strategies.extend(
            [
                _evaluation_mapping[_design_method](_design_method)
                for _design_method in self.vr_config.design_methods
                if _design_method in _evaluation_mapping.keys()
            ]
        )

        logging.info("Stap 3: Bepaling maatregelen op trajectniveau afgerond")

        return _results_optimization
