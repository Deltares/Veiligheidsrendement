from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.section_as_input import SectionAsInput


class AggregateCombinationsController:
    def __init__(self, section: SectionAsInput) -> None:
        self._section = section

    @staticmethod
    def _create_aggregates(
        sh_combinations: list[CombinedMeasure], sg_combinations: list[CombinedMeasure]
    ) -> list[AggregatedMeasureCombination]:
        _aggr_meas_comb = []
        for _sh in sh_combinations:
            # Year and primary measure type should match
            _sg_combinations = filter(
                lambda x: x.primary.year == _sh.primary.year
                and x.primary.measure_type == _sh.primary.measure_type,
                sg_combinations,
            )
            for _sg in _sg_combinations:
                _aggr_meas_comb.append(
                    AggregatedMeasureCombination(_sh, _sg, _sh.primary.year)
                )
        return _aggr_meas_comb

    def aggregate(self) -> list[AggregatedMeasureCombination]:
        return self._create_aggregates(
            self._section.sh_combinations, self._section.sg_combinations
        )
