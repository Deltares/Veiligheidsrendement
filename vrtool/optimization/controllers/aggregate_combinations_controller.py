from itertools import product

from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.combined_measures.combined_measure_factory import (
    CombinedMeasureFactory,
)
from vrtool.optimization.measures.combined_measures.sg_combined_measure import (
    SgCombinedMeasure,
)
from vrtool.optimization.measures.combined_measures.sh_combined_measure import (
    ShCombinedMeasure,
)
from vrtool.optimization.measures.combined_measures.shsg_combined_measure import (
    ShSgCombinedMeasure,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput


class AggregateCombinationsController:
    """
    Controller responsable of creating a collection of `AggregatedCombinedMeasure`.
    This controller is also responsible of creating the "additional" `ShSgCombinedMeasure`
    when required by an `AggregatedCombinedMeasure`.
    """

    def __init__(self, section: SectionAsInput) -> None:
        self._section = section

    def _get_aggregated_measure_id(
        self,
        sh_comb: ShCombinedMeasure,
        sg_comb: SgCombinedMeasure,
        shsg_comb: ShSgCombinedMeasure | None,
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
        self, sh_combination: ShCombinedMeasure, sg_combination: SgCombinedMeasure
    ) -> AggregatedMeasureCombination:
        _shsg_combined_measure = CombinedMeasureFactory.get_shsg_combined_measure(
            self._section.sh_sg_measures, sh_combination, sg_combination
        )
        return AggregatedMeasureCombination(
            sh_combination=sh_combination,
            sg_combination=sg_combination,
            shsg_combination=_shsg_combined_measure,
            measure_result_id=self._get_aggregated_measure_id(
                sh_combination, sg_combination, _shsg_combined_measure
            ),
            year=sh_combination.primary.year,
        )

    def aggregate(self) -> list[AggregatedMeasureCombination]:
        """
        Creates all possible aggregations based on the section's
        Sh and Sg combinations (`CombinedMeasureBase`)

        Returns:
            list[AggregatedMeasureCombination]:
            Resulting list of aggregated combinations.
        """

        def combinations_can_be_aggregated(
            combinations: tuple[CombinedMeasureBase, CombinedMeasureBase]
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
