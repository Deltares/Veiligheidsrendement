from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.vertical_piping_measure_calculator_base import (
    VerticalPipingMeasureCalculatorBase,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.vertical_piping_measure_calculator_protocol import (
    VerticalPipingMeasureCalculatorProtocol,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class VerticalGeotextileMeasureCalculator(
    VerticalPipingMeasureCalculatorBase, VerticalPipingMeasureCalculatorProtocol
):
    """
    Applied when `2m <= cover_layer_thickness < 4m`.
    * It reduces the `pf_piping` with a factor `1000`.
    * It has a price of `1700€/m`.
    """

    @property
    def pf_piping_reduction_factor(self) -> float:
        return 1000

    def calculate_cost(self, unit_costs: MeasureUnitCosts) -> float:
        return unit_costs.vertical_geotextile * self.dike_section.Length

    def calculate_reliability(self) -> SectionReliability:
        return self._get_configured_section_reliability()
