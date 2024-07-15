from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.vertical_piping_measure_calculator_base import (
    VerticalPipingMeasureCalculatorBase,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.vertical_piping_measure_calculator_protocol import (
    VerticalPipingMeasureCalculatorProtocol,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class HeavescreenMeasureCalculator(
    VerticalPipingMeasureCalculatorBase, VerticalPipingMeasureCalculatorProtocol
):
    """
    Applied when `4m < cover_layer_thickness`.
    * It reduces the `pf_piping` with a factor `1000`.
    * The unit cost:
        * It is assumed to be lower than that from the unanchored sheetpile.
        * It is expressed per m2, so we need to calculate the vertical length of the screen.
        * The assumption is that it should go 6m below the cover_layer, so `l_screen = cover_layer_thickness + 6m`.
    """

    def calculate_cost(self, unit_costs: MeasureUnitCosts) -> float:
        _vertical_length = self.dike_section.cover_layer_thickness + 6
        return unit_costs.heavescreen * _vertical_length * self.dike_section.Length

    def calculate_reliability(self) -> SectionReliability:
        return self._get_configured_section_reliability()
