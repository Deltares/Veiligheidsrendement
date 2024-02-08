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
    def sh_measures(self):
        return [m for m in self.measures if isinstance(m, ShMeasure)]

    @property
    def sg_measures(self):
        return [m for m in self.measures if not isinstance(m, SgMeasure)]
