from itertools import product

from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.section_as_input import SectionAsInput


class AggregateCombinationsController:
    def __init__(self, section: SectionAsInput) -> None:
        self._section = section

    def _create_aggregates(
        self,
        sh_combinations: list[CombinedMeasure],
        sg_combinations: list[CombinedMeasure],
    ) -> list[AggregatedMeasureCombination]:
        def primaries_match(
            aggregation: tuple[CombinedMeasure, CombinedMeasure]
        ) -> bool:
            # Check if the primary measures in both commbinations match
            _sh_comb, _sg_comb = aggregation
            return (
                _sh_comb.primary.year == _sg_comb.primary.year
                and _sh_comb.primary.measure_type == _sg_comb.primary.measure_type
            )

        def make_aggregate(
            aggregation: tuple[CombinedMeasure, CombinedMeasure]
        ) -> AggregatedMeasureCombination:
            _sh_comb, _sg_comb = aggregation
            return AggregatedMeasureCombination(
                _sh_comb, _sg_comb, _sh_comb.primary.year
            )

        return list(
            map(
                make_aggregate,
                filter(primaries_match, product(sh_combinations, sg_combinations)),
            )
        )

    def aggregate(self) -> list[AggregatedMeasureCombination]:

        return self._create_aggregates(
            self._section.sh_combinations, self._section.sg_combinations
        )
