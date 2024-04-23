from vrtool.decision_making.measures.standard_measures.vertical_piping_measures.vertical_piping_measure_calculator_base import (
    VerticalPipingMeasureCalculatorBase,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measures.vertical_piping_measure_calculator_protocol import (
    VerticalPipingMeasureCalculatorProtocol,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class HeavescreenMeasureCalculator(
    VerticalPipingMeasureCalculatorBase, VerticalPipingMeasureCalculatorProtocol
):
    def calculate_cost(self, unit_costs: dict[str, float]) -> float:
        return unit_costs["Heavescreen"] * self.dike_section.Length

    def calculate_reliability(self) -> SectionReliability:
        return self._get_configured_section_reliability()
