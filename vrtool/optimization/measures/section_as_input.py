from dataclasses import dataclass

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

    @property
    def sh_measures(self) -> list[ShMeasure]:
        return list(filter(lambda x: isinstance(x, ShMeasure), self.measures))

    @property
    def sg_measures(self) -> list[SgMeasure]:
        return list(filter(lambda x: isinstance(x, SgMeasure), self.measures))
