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
    secondary: MeasureAsInputProtocol

    @property
    def lcc(self) -> float:
        return self.primary.lcc + self.secondary.lcc

    @property
    def mechanism_year_collection(self) -> MechanismPerYearProbabilityCollection:
        return self.primary.mechanism_year_collection.combine(
            self.secondary.mechanism_year_collection
        )
