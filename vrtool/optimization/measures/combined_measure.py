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

    def __init__(self, primary_measure: MeasureAsInputProtocol, secondary_measure: MeasureAsInputProtocol) -> None:
        self.primary = primary_measure
        self.secondary = secondary_measure
        self.cost = [primary_measure.cost, secondary_measure.cost]
        self.lcc = primary_measure.lcc + secondary_measure.lcc
        self.mechanism_year_collection = primary_measure.mechanism_year_collection.combine(secondary_measure.mechanism_year_collection)
