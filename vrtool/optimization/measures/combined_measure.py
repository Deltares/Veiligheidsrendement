from __future__ import annotations

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

    @property
    def measure_class(self) -> str:
        if self.secondary is not None:
            return "combined"
        return self.primary.combine_type.get_old_name()

    @property
    def dcrest(self) -> float:
        return self.primary.dcrest

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
        if self.secondary is not None:
            return [self.primary.year, self.secondary.year]
        return self.primary.year

    @property
    def beta_target(self) -> float:
        if isinstance(self.primary, ShMeasure):
            return self.primary.transition_level
        return -999

    @property
    def yesno(self) -> int | str:
        if self.primary.measure_type in [
            MeasureTypeEnum.VERTICAL_GEOTEXTILE,
            MeasureTypeEnum.DIAPHRAGM_WALL,
            MeasureTypeEnum.STABILITY_SCREEN,
        ]:
            return "yes"
        if self.secondary is not None and self.secondary.measure_type in [
            MeasureTypeEnum.VERTICAL_GEOTEXTILE,
            MeasureTypeEnum.DIAPHRAGM_WALL,
            MeasureTypeEnum.STABILITY_SCREEN,
        ]:
            return "yes"
        return -999

    @property
    def combined_id(self) -> int | str:
        if self.secondary is not None:
            return (
                f"{self.primary.measure_type.value}+{self.secondary.measure_type.value}"
            )
        return self.primary.measure_type.value

    @property
    def combined_measure_type(self) -> str:
        if self.secondary is not None:
            return f"{self.primary.measure_type.get_old_name()}+{self.secondary.measure_type.get_old_name()}"
        return self.primary.measure_type.get_old_name()

    @property
    def combined_db_index(self) -> list[int]:
        if self.secondary is not None:
            return [self.primary.measure_result_id, self.secondary.measure_result_id]
        return [self.primary.measure_result_id]

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
