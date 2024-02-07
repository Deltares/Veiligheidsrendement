from dataclasses import dataclass

from vrtool.optimization.measures.measure_as_input_protocol import MeasureAsInputProtocol
from mechanism_per_year_probability_collection import MechanismPerYearProbabilityCollection

@dataclass
class CombinedMeasure:
    primary: MeasureAsInputProtocol
    secondary: MeasureAsInputProtocol
    mechanism_year_collection: MechanismPerYearProbabilityCollection
