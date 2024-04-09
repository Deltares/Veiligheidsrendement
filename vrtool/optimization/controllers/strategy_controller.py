import logging

from tqdm import tqdm

from vrtool.decision_making.strategies import GreedyStrategy, TargetReliabilityStrategy
from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.controllers.aggregate_combinations_controller import (
    AggregateCombinationsController,
)
from vrtool.optimization.controllers.combine_measures_controller import (
    CombineMeasuresController,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.strategy_input.strategy_input import StrategyInput
from vrtool.optimization.strategy_input.strategy_input_protocol import (
    StrategyInputProtocol,
)


class StrategyController:
    _method: str
    _vrtool_config: VrtoolConfig
    _section_measures_input: list[SectionAsInput]

    def __init__(self, section_measures_input: list[SectionAsInput]) -> None:
        self._section_measures_input = section_measures_input

    def combine(self) -> None:
        """
        Combines the measures for each section.
        """
        logging.info("Combineren van maatregelen per dijkvak.")
        for _section in tqdm(
            self._section_measures_input,
            desc="Aantal dijkvakken gecombineerd: ",
            unit="dijkvak",
        ):
            _combine_controller = CombineMeasuresController(_section)
            _section.combined_measures = _combine_controller.combine()

    def aggregate(self) -> None:
        """
        Aggregates combinations of measures for each section.
        """
        logging.info("Aggregeren van maatregelen per dijkvak.")
        for _section in tqdm(
            self._section_measures_input,
            desc="Aantal dijkvakken geaggregeerd: ",
            unit="dijkvak",
        ):
            _aggregate_controller = AggregateCombinationsController(_section)
            _section.aggregated_measure_combinations = _aggregate_controller.aggregate()

    def get_evaluate_input(
        self, strategy_type: type[StrategyProtocol]
    ) -> StrategyInputProtocol:
        """
        Get the input for the evaluation of the strategy.
        """
        if strategy_type in [GreedyStrategy, TargetReliabilityStrategy]:
            return StrategyInput.from_section_as_input_collection(
                self._section_measures_input
            )
        raise ValueError(f"Strategy type {strategy_type} not implemented yet.")

    def set_investment_year(self) -> None:
        """
        Set investment year for all sections
        """
        logging.info(
            "Toevoegen configuratie voor investeringsjaren toe aan maatregelen per dijkvak."
        )
        for _section_as_input in tqdm(
            self._section_measures_input, desc="Aantal dijkvakken: ", unit="dijkvak"
        ):
            _section_as_input.update_measurelist_with_investment_year()
