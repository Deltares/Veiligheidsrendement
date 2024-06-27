import logging
from itertools import product

from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.section_as_input import SectionAsInput


class AggregateCombinationsController:
    def __init__(self, section: SectionAsInput) -> None:
        self._section = section

    def _get_shsg_combined_measure(
        self, sh_comb: CombinedMeasure, sg_comb: CombinedMeasure
    ) -> CombinedMeasure | None:
        _found_shsg_measures = [
            _shsg_measure
            for _shsg_measure in self._section.sh_sg_measures
            if _shsg_measure.matches_with_sh_sg_measure(sh_comb, sg_comb)
        ]
        if not _found_shsg_measures:
            return None

        if len(_found_shsg_measures) > 1:
            logging.warning(
                "More than one `ShSgMeasure` found for combination of primary measure results (%s, %s). Using only the first one found.",
                sh_comb.primary.measure_result_id,
                sg_comb.primary.measure_result_id,
            )
        _shsg_measure = _found_shsg_measures[0]
        return CombinedMeasure(
            primary=_shsg_measure,
            secondary=None,
            mechanism_year_collection=_shsg_measure.mechanism_year_collection,
        )

    def _get_aggregated_measure_id(
        self,
        sh_comb: CombinedMeasure,
        sg_comb: CombinedMeasure,
        shsg_comb: CombinedMeasure | None,
    ) -> int:
        # Find the aggregated Sh/Sg measure result id
        if sh_comb.primary.measure_result_id == sg_comb.primary.measure_result_id:
            return sh_comb.primary.measure_result_id
        if sh_comb.primary.dcrest == 0:
            return sg_comb.primary.measure_result_id
        if sg_comb.primary.dberm == 0:
            return sh_comb.primary.measure_result_id

        if shsg_comb is None:
            _sh_str = f"Sh ({sh_comb.primary.measure_result_id})"
            _sg_str = f"Sg ({sg_comb.primary.measure_result_id})"
            raise ValueError(
                f"Geen `MeasureResult.id` gevonden tussen gecombineerd (primary) maatregelen met `MeasureResult.id`: {_sh_str} en {_sg_str}."
            )
        return shsg_comb.primary.measure_result_id

    def _make_aggregate(
        self, sh_combination: CombinedMeasure, sg_combination: CombinedMeasure
    ) -> AggregatedMeasureCombination:
        _shsg_combined_measure = self._get_shsg_combined_measure(
            sh_combination, sg_combination
        )
        return AggregatedMeasureCombination(
            sh_combination=sh_combination,
            sg_combination=sg_combination,
            sh_sg_combination=_shsg_combined_measure,
            measure_result_id=self._get_aggregated_measure_id(
                sh_combination, sg_combination, _shsg_combined_measure
            ),
            year=sh_combination.primary.year,
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
