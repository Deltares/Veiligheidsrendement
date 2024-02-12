from dataclasses import dataclass
from typing import Optional

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
    combined_measures: Optional[list[CombinedMeasure]]
    aggregated_measure_combinations: Optional[list[AggregatedMeasureCombination]]

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
                lambda x: isinstance(x, measure_class), self.combined_measures.primary
            )
        )

    @property
    def sh_combinations(self) -> list[CombinedMeasure]:
        return self.get_combinations_by_class(ShMeasure)

    @property
    def sg_combinations(self) -> list[CombinedMeasure]:
        return self.get_combinations_by_class(SgMeasure)
