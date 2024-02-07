from dataclasses import dataclass

from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)


@dataclass
class SectionAsInput
    section_name: str
    traject_name: str
    measures: list[MeasureAsInputProtocol]
