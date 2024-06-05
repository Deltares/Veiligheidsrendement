from dataclasses import dataclass

from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.orm.models.measure_result.measure_result import MeasureResult


@dataclass
class MeasureAsInputImporterData:
    measure_as_input_type: type[MeasureAsInputProtocol]
    concrete_parameters: list[str]
    measure_result: MeasureResult
    investment_years: list[int]
    discount_rate: float
