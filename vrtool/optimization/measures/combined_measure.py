from __future__ import annotations

import math
from dataclasses import dataclass

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


@dataclass
class CombinedMeasure:
    primary: MeasureAsInputProtocol
    secondary: MeasureAsInputProtocol | None
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    # Legacy index for mapping back to the old structure for evaluate
    sequence_nr: int = None

    def is_base_measure(self) -> bool:
        """
        Determines whether this `CombinedMeasure` could be considered
        as a base measure (usually when `dberm` / `dcrest` equal to 0).

        Returns:
            bool: True when its primary measure is an initial measure.
        """
        return self.primary.is_base_measure()

    @property
    def cost(self) -> float:
        """
        Combined measure cost, can be used to compute the `LCC` of
        an aggregated measure.

        Returns:
            float: Total (bruto) cost of the combined measures.
        """
        if self.secondary:
            return self.primary.cost + self.secondary.cost
        return self.primary.cost

    def compares_to(self, other: "CombinedMeasure") -> bool:
        """
        Compares this instance of a 'CombinedMeasure' with another one.
        This method could be also implemented as an overriding of the
        'equal' operator ('__eq__').

        Args:
            other (CombinedMeasure): Other combined measure to compare.

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

    @classmethod
    def from_input(
        cls,
        primary: MeasureAsInputProtocol,
        secondary: MeasureAsInputProtocol | None,
        initial_assessment: MechanismPerYearProbabilityCollection,
        sequence_nr: int,
    ) -> CombinedMeasure:
        """
        Create a combined measure from input

        Args:
            primary (MeasureAsInputProtocol): The primary measure
            secondary (MeasureAsInputProtocol | None): The secondary measure
            initial_assessment (MechanismPerYearProbabilityCollection): The initial assessment
            sequence_nr (int): The sequence nr of the combination in the list of Sg- or Sh-combinations

        Returns:
            CombinedMeasure: The combined measure
        """
        _mech_year_coll = primary.mechanism_year_collection
        if secondary:
            _mech_year_coll = MechanismPerYearProbabilityCollection.combine(
                primary.mechanism_year_collection,
                secondary.mechanism_year_collection,
                initial_assessment,
            )

        return cls(
            primary=primary,
            secondary=secondary,
            mechanism_year_collection=_mech_year_coll,
            sequence_nr=sequence_nr,
        )
