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
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    cost: list[float]
    lcc: float

    def __init__(self, primaryMeasure: MeasureAsInputProtocol, secondaryMeasure: MeasureAsInputProtocol) -> None:
        self.primary = primaryMeasure
        self.secondary = secondaryMeasure
        self.cost = [primaryMeasure.cost, secondaryMeasure.cost]
        self.lcc = primaryMeasure.lcc + secondaryMeasure.lcc
        self.mechanism_year_collection = primaryMeasure.mechanism_year_collection.combine(secondaryMeasure.mechanism_year_collection)
