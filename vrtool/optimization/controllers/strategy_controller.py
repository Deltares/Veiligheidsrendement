from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.controllers.aggregate_combinations_controller import (
    AggregateCombinationsController,
)
from vrtool.optimization.controllers.combine_measures_controller import (
    CombineMeasuresController,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.strategy_input import StrategyInput


class StrategyController:
    _method: str
    _vrtool_config: VrtoolConfig
    _section_measures_input: list[SectionAsInput]

    def __init__(self, method: str, vrtool_config: VrtoolConfig) -> None:
        self._method = method
        self._vrtool_config = vrtool_config
        self._section_measures_input = []

    def combine(self) -> None:
        """
        Combines the measures for each section.
        """
        for _section in self._section_measures_input:
            _combine_controller = CombineMeasuresController(_section)
            _section.combined_measures = _combine_controller.combine()

    def aggregate(self) -> None:
        """
        Aggregates combinations of measures for each section.
        """
        for _section in self._section_measures_input:
            _aggregate_controller = AggregateCombinationsController(_section)
            _section.aggregated_measure_combinations = _aggregate_controller.aggregate()

    def get_evaluate_input(self) -> StrategyInput:
        """
        Get the input for the evaluation of the strategy.
        """
        return StrategyInput.from_section_as_input(self._section_measures_input)

    def set_investment_year(self) -> None:
        """
        Set investment year for all sections
        """
        logging.info("Voeg configuratie voor investeringsjaren toe aan maatregelen per dijkvak.")
        for _section_as_input in tqdm(self._section_measures_input, desc="Aantal dijkvakken: ", unit="dijkvak"):
            _section_as_input.update_measurelist_with_investment_year()
