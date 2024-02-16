from dataclasses import dataclass

from vrtool.optimization.measures.combined_measure import CombinedMeasure


@dataclass
class AggregatedMeasureCombination:
    sh_combination: CombinedMeasure
    sg_combination: CombinedMeasure
    year: int

    @property
    def lcc(self):
        return self.sh_combination.lcc + self.sg_combination.lcc
