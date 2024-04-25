from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.vertical_piping_measure_calculator_base import (
    VerticalPipingMeasureCalculatorBase,
)
from vrtool.decision_making.measures.standard_measures.vertical_piping_measure_calculators.vertical_piping_measure_calculator_protocol import (
    VerticalPipingMeasureCalculatorProtocol,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class FallbackMeasureCalculator(
    VerticalPipingMeasureCalculatorBase, VerticalPipingMeasureCalculatorProtocol
):
    """
    This calculator is meant to be used when `dcover > 6`.
    We then assume minimal costs as the piping probability is minimal.
    """

    @property
    def pf_piping_reduction_factor(self) -> float:
        """
        Gets the default reduction factor for `pf_piping` ( `P_solution` ).
        This property can be overriden when inheriting from this class.

        Returns:
            float: reduction value.
        """
        return 1

    def calculate_cost(self, unit_costs: MeasureUnitCosts) -> float:
        return 0

    def calculate_reliability(self) -> SectionReliability:
        return self._get_configured_section_reliability()
