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

    def check_primary_measure_result_id_and_year(self, primary_sh, primary_sg):
        if (self.sh_combination.primary.measure_result_id == primary_sh.measure_result_id) and (self.sg_combination.primary.measure_result_id == primary_sg.measure_result_id):
            if (self.sh_combination.primary.year == primary_sh.year) and (self.sg_combination.primary.year == primary_sg.year):
                return True
            else:
                return False
        else:
            return False