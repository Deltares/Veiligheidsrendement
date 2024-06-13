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
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


@dataclass
class CombinedMeasure:
    primary: MeasureAsInputProtocol
    secondary: MeasureAsInputProtocol | None
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    # Legacy index for mapping back to the old structure for evaluate
    sequence_nr: int = None

    def is_initial_measure(self) -> bool:
        """
        Determines whether this `CombinedMeasure` could be considered
        as an initial measure (usually when `dberm` / `dcrest` equal to 0).

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

    @property
    def measure_class(self) -> str:
        if self.secondary:
            return "combined"
        return self.primary.combine_type.legacy_name

    @property
    def dcrest(self) -> float:
        if isinstance(self.primary, ShMeasure):
            return self.primary.dcrest
        return -999

    @property
    def dberm(self) -> float:
        if isinstance(self.primary, SgMeasure):
            return self.primary.dberm
        return -999

    @property
    def transition_level(self) -> float:
        if isinstance(self.primary, ShMeasure):
            return self.primary.transition_level
        return -999

    @property
    def year(self) -> int | list[int]:
        if self.secondary:
            return [self.primary.year, self.secondary.year]
        return self.primary.year

    @property
    def beta_target(self) -> float:
        if isinstance(self.primary, ShMeasure):
            return self.primary.transition_level
        return -999

    @property
    def yesno(self) -> int | str:
        _accepted_measure_types = [
            MeasureTypeEnum.VERTICAL_PIPING_SOLUTION,
            MeasureTypeEnum.DIAPHRAGM_WALL,
            MeasureTypeEnum.STABILITY_SCREEN,
        ]
        if self.primary.measure_type in _accepted_measure_types:
            return "yes"
        if self.secondary and self.secondary.measure_type in _accepted_measure_types:
            return "yes"
        return -999

    @property
    def combined_id(self) -> str:
        if self.secondary:
            return (
                f"{self.primary.measure_type.value}+{self.secondary.measure_type.value}"
            )
        return self.primary.measure_type.value

    @property
    def combined_measure_type(self) -> str:
        if self.secondary:
            return f"{self.primary.measure_type.legacy_name}+{self.secondary.measure_type.legacy_name}"
        return self.primary.measure_type.legacy_name

    @property
    def combined_db_index(self) -> list[int]:
        if self.secondary:
            return [self.primary.measure_result_id, self.secondary.measure_result_id]
        return [self.primary.measure_result_id]

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

        return (
            self.primary.year == other.primary.year
            and self.primary.measure_type == other.primary.measure_type
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
