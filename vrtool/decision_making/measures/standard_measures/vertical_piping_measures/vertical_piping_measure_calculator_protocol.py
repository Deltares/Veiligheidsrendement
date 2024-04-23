from typing import Protocol, runtime_checkable

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.section_reliability import SectionReliability


@runtime_checkable
class VerticalPipingMeasureCalculatorProtocol(Protocol):
    traject_info: DikeTrajectInfo
    dike_section: DikeSection

    def is_valid(self) -> bool:
        """
        Validates whether the current state of this calculator fulfills
         the requirements to generate costs or reliability.

        Returns:
            bool: Calculator is in a valid state.
        """

    def calculate_cost(self, unit_costs: dict[str, float]) -> float:
        """
        Calculates the costs associated to applying a specific
        `VerticalPipingMeasureCalculatorProtocol`.

        Args:
            unit_costs (dict[str, float]): Reference from where to extract the unitarian costs.

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
