from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.controllers.aggregate_combinations_controller import (
    AggregateCombinationsController,
)
from vrtool.optimization.controllers.combine_measures_controller import (
    CombineMeasuresController,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput


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
