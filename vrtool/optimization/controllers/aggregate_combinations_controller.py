from itertools import product

from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.section_as_input import SectionAsInput


class AggregateCombinationsController:
    def __init__(self, section: SectionAsInput) -> None:
        self._section = section

    def _get_aggregated_measure_id(
        self, sh_comb: CombinedMeasure, sg_comb: CombinedMeasure
    ) -> int:
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
                if m.matches_with_sh_sg_measure(sh_comb, sg_comb)
            ),
            0,
        )

    def _make_aggregate(
        self, sh_combination: CombinedMeasure, sg_combination: CombinedMeasure
    ) -> AggregatedMeasureCombination:
        return AggregatedMeasureCombination(
            sh_combination,
            sg_combination,
            self._get_aggregated_measure_id(sh_combination, sg_combination),
            sh_combination.primary.year,
        )

    def aggregate(self) -> list[AggregatedMeasureCombination]:
        """
        Creates all possible aggregations based on the section's
        Sh and Sg combinations (`CombinedMeasure`)

        Returns:
            list[AggregatedMeasureCombination]:
            Resulting list of aggregated combinations.
        """

        def combinations_can_be_aggregated(
            combinations: tuple[CombinedMeasure, CombinedMeasure]
        ) -> bool:
            return combinations[0].compares_to(combinations[1])

        # Filter combinations that can be aggregated.
        _combinations_to_aggregate = filter(
            combinations_can_be_aggregated,
            product(self._section.sh_combinations, self._section.sg_combinations),
        )

        # Create aggregations.
        return [
            self._make_aggregate(_c_sh, _c_sg)
            for _c_sh, _c_sg in _combinations_to_aggregate
        ]
