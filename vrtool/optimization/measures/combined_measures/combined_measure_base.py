from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


@dataclass(kw_only=True)
class CombinedMeasureBase:
    """
    Base class to represent the combination between two measures
    where the most important (`primary`) will have the leading
    base costs over the auxiliar (`secondary`) one.
    """

    primary: MeasureAsInputProtocol
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    # Legacy index for mapping back to the old structure for evaluate
    sequence_nr: int = None

    @property
    def lcc(self) -> float:
        """
        Calculates the "life cycle cost" of this combined measure
        based on their respectives `cost` values and discount rate / year.

        Returns:
            float: The total (bruto) lcc of this combined measure.
        """
        return self._get_primary_lcc() + self._get_secondary_lcc()

    @property
    @abstractmethod
    def _base_cost(self) -> float:
        """
        The base cost of this `CombinedMeasureBase` instance.

        Returns:
            float: The (bruto) cost of this combined measure.
        """

    def is_base_measure(self) -> bool:
        """
        Determines whether this `CombinedMeasureBase` could be considered
        as a base measure (usually when `dberm` / `dcrest` equal to 0).

        Returns:
            bool: True when its primary measure is an initial measure.
        """
        return self.primary.is_base_measure()

    def _calculate_lcc(
        self, measure_as_input: MeasureAsInputProtocol, base_cost: float
    ) -> float:
        return (measure_as_input.cost - base_cost) / (
            (1 + measure_as_input.discount_rate) ** measure_as_input.year
        )

    def _get_primary_lcc(self) -> float:
        # Calculate the costs for the primary measure.
        return self._calculate_lcc(self.primary, self._base_cost)

    @abstractmethod
    def _get_secondary_lcc(self) -> float:
        pass

    def compares_to(self, other: "CombinedMeasureBase") -> bool:
        """
        Compares this instance of a 'CombinedMeasureBase' with another one.
        This method could be also implemented as an overriding of the
        'equal' operator ('__eq__').

        Args:
            other (CombinedMeasureBase): Other combined measure to compare.

        Returns:
            bool: Whether both combined measures can be conisered as matching.
        """
        # Check if the primary measures in both combinations match
        def compatible_l_stab_screen() -> bool:
            if math.isnan(self.primary.l_stab_screen) or math.isnan(
                other.primary.l_stab_screen
            ):
                return True
            return self.primary.l_stab_screen == other.primary.l_stab_screen

        def compatible_measure_type() -> bool:
            if self.primary.measure_type != MeasureTypeEnum.CUSTOM:
                return self.primary.measure_type == other.primary.measure_type
            return (
                self.primary.measure_type == other.primary.measure_type
                and self.primary.measure_result_id == other.primary.measure_result_id
            )

        return (
            self.primary.year == other.primary.year
            and compatible_measure_type()
            and compatible_l_stab_screen()
        )
