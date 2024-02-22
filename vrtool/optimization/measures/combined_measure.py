from __future__ import annotations

from dataclasses import dataclass

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

    @property
    def combined_years(self) -> list[int]:
        """Get the years of the measure."""
        if not self.secondary:
            return [self.primary.year]
        return [self.primary.year, self.secondary.year]

    @property
    def combined_id(self) -> str:
        """
        Gets an `ID` representing this combined measure.
        When no `secondary` measure is present the `measure_result_id`
          from the primary will be used.
        When a `secondary` measure is present then both `measure_result_id`
        are used joined by the `+` symbol.

        Returns:
            str: The combined measure's `ID`.
        """
        if not self.secondary:
            return self.primary.measure_result_id
        return "{}+{}".format(
            self.primary.measure_result_id, self.secondary.measure_result_id
        )

    @property
    def name(self) -> str:
        """
        Gets a name representing this combined measure.
        When no `secondary` measure is present the `measure_type`
          from the primary will be used.
        When a `secondary` measure is present then both `measure_type`
        are used joined by the `+` symbol.

        Returns:
            str: The combined measure's name.
        """
        if not self.secondary:
            return self.primary.measure_type.name
        return "{}+{}".format(
            self.primary.measure_type.name, self.secondary.measure_type.name
        )

    @property
    def class_name(self) -> str:
        """
        Gets a class name representing this combined measure.
        When no `secondary` measure is present the `combine_type`
          from the primary will be used.
        When a `secondary` measure is present then both `combine_type`
        are used joined by the `+` symbol.

        Returns:
            str: The combined measure's class name.
        """
        if not self.secondary:
            return self.primary.combine_type.get_old_name()
        return "combined"

    @property
    def lcc(self) -> float:
        if self.secondary is not None:
            return self.primary.lcc + self.secondary.lcc
        return self.primary.lcc

    @classmethod
    def from_input(
        cls,
        primary: MeasureAsInputProtocol,
        secondary: MeasureAsInputProtocol | None,
    ) -> CombinedMeasure:
        _mech_year_coll = primary.mechanism_year_collection
        if secondary is not None:
            _mech_year_coll = MechanismPerYearProbabilityCollection.combine(
                primary.mechanism_year_collection,
                secondary.mechanism_year_collection,
            )

        return cls(
            primary=primary,
            secondary=secondary,
            mechanism_year_collection=_mech_year_coll,
        )
