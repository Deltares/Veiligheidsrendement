from dataclasses import dataclass, field
from typing import Optional

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


@dataclass
class SectionAsInput:
    section_name: str
    traject_name: str
    measures: list[MeasureAsInputProtocol]
    combined_measures: list[CombinedMeasure] = field(
        default_factory=list[CombinedMeasure]
    )  # TODO do we need this in SectionAsInput or can it be volatile?
    aggregated_measure_combinations: Optional[list[AggregatedMeasureCombination]] = (
        field(default_factory=list[AggregatedMeasureCombination])
    )

    def get_measures_by_class(
        self,
        measure_class: type[MeasureAsInputProtocol],
    ) -> list[MeasureAsInputProtocol]:
        """
        Get the measures for a section based on class of measure (Sg/Sh).

        Args:
            measure_class (type[MeasureAsInputProtocol]): Class of measure.

        Returns:
            list[MeasureAsInputProtocol]: Measures of the class.
        """
        return list(filter(lambda x: isinstance(x, measure_class), self.measures))

    @property
    def sh_measures(self) -> list[MeasureAsInputProtocol]:
        return self.get_measures_by_class(ShMeasure)

    @property
    def sg_measures(self) -> list[MeasureAsInputProtocol]:
        return self.get_measures_by_class(SgMeasure)

    def get_combinations_by_class(
        self, measure_class: type[MeasureAsInputProtocol]
    ) -> list[CombinedMeasure]:
        """
        Get the combinations of measures for a section
        based on the class of measure of the primary measure (Sg/Sh).

        Args:
            measure_class (type[MeasureAsInputProtocol]): Class of measure.

        Returns:
            list[CombinedMeasure]: Combined measures of the class.
        """
        return list(
            filter(
                lambda x: isinstance(x.primary, measure_class), self.combined_measures
            )
        )

    @property
    def sh_combinations(self) -> list[CombinedMeasure]:
        return self.get_combinations_by_class(ShMeasure)

    @property
    def sg_combinations(self) -> list[CombinedMeasure]:
        return self.get_combinations_by_class(SgMeasure)

    @property
    def max_year(self) -> int:
        """
        The maximum year for the section

        Returns:
            int: The maximum year
        """
        if not self.measures:
            return 0
        # Get the max year for all measures for a random mechanism
        return max(
            year
            for meas in self.measures
            for year in meas.mechanism_year_collection.get_years(
                self.measures[0].get_allowed_mechanisms()[0]
            )
        )

    @property
    def mechanisms(self) -> set[MechanismEnum]:
        """
        All mechanisms for the section

        Returns:
            set[MechanismEnum]: Set of mechanisms
        """
        if not self.measures:
            return set()
        return set(
            mech
            for meas in self.measures
            for mech in meas.mechanism_year_collection.get_mechanisms()
        )
