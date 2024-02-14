from dataclasses import dataclass, field

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

    def get_measures_by_class(
        self,
        measure_type: type[MeasureAsInputProtocol],
    ) -> list[MeasureAsInputProtocol]:
        """
        Get the measures for a section based on class of measure (Sg/Sh).
        Args:
            measure_class (type[MeasureAsInputProtocol]): Class of measure.
        Returns:
            list[MeasureAsInputProtocol]: Measures of the class.
        """
        return list(filter(lambda x: isinstance(x, measure_type), self.measures))

    @property
    def sh_measures(self) -> list[MeasureAsInputProtocol]:
        return self.get_measures_by_class(ShMeasure)

    @property
    def sg_measures(self) -> list[MeasureAsInputProtocol]:
        return self.get_measures_by_class(SgMeasure)
