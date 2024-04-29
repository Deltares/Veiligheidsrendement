from typing import Protocol, runtime_checkable

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.section_reliability import SectionReliability


@runtime_checkable
class VerticalPipingMeasureCalculatorProtocol(Protocol):
    traject_info: DikeTrajectInfo
    dike_section: DikeSection

    def calculate_cost(self, unit_costs: MeasureUnitCosts) -> float:
        """
        Calculates the costs associated to applying a specific
        `VerticalPipingMeasureCalculatorProtocol`.

        Args:
            unit_costs (MeasureUnitCosts): (Dataclass) Instance containing unitarian costs.

        Returns:
            float: the total cost (in Euros).
        """

    def calculate_reliability(self) -> SectionReliability:
        """
        Calculates the reliability when applying a specific
        `VerticalPipingMeasureCalculatorProtocol`.

        Returns:
            SectionReliability: The calculated reliability.
        """
