from dataclasses import dataclass

from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure

@dataclass
class AggregatedMeasureCombination:
    sh_combination: CombinedMeasure
    sg_combination: CombinedMeasure
    measure_result_id: int
    year: int

    @property
    def lcc(self):
        return self.sh_combination.lcc + self.sg_combination.lcc
