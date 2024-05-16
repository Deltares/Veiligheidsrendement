import math
from itertools import product

from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure


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
            # Check if the primary measures in both combinations match
            _sh_comb, _sg_comb = aggregation
            return (
                _sh_comb.primary.year == _sg_comb.primary.year
                and _sh_comb.primary.measure_type == _sg_comb.primary.measure_type
            )

        def get_aggregated_measure_id(
            sh_comb: CombinedMeasure, sg_comb: CombinedMeasure
        ) -> int:
            def is_matching_stab_length(sh_sg_length: float, sg_comb_length: float):
                if math.isnan(sh_sg_length) and math.isnan(sg_comb_length):
                    return True
                return sh_sg_length == sg_comb_length

            def is_matching_sh_sg_measure(sh_sg_measure: ShSgMeasure) -> bool:
                return (
                    sh_sg_measure.dcrest == sh_comb.primary.dcrest
                    and sh_sg_measure.dberm == sg_comb.primary.dberm
                    and is_matching_stab_length(
                        sh_sg_measure.l_stab_screen, sg_comb.primary.l_stab_screen
                    )
                    and sh_sg_measure.measure_type == sh_comb.primary.measure_type
                )

            # Find the aggregated Sh/Sg measure result id
            if sh_comb.primary.measure_result_id == sg_comb.primary.measure_result_id:
                return sh_comb.primary.measure_result_id
            if sh_comb.primary.dcrest == 0:
                return sg_comb.primary.measure_result_id
            if sg_comb.primary.dberm == 0:
                return sh_comb.primary.measure_result_id
            return next(
                (
                    m.measure_result_id
                    for m in self._section.sh_sg_measures
                    if is_matching_sh_sg_measure(m)
                ),
                0,
            )

        def make_aggregate(
            aggregation: tuple[CombinedMeasure, CombinedMeasure]
        ) -> AggregatedMeasureCombination:
            _sh_comb, _sg_comb = aggregation
            return AggregatedMeasureCombination(
                _sh_comb,
                _sg_comb,
                get_aggregated_measure_id(_sh_comb, _sg_comb),
                _sh_comb.primary.year,
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
