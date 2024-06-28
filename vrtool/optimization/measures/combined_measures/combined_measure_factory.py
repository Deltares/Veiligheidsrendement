import logging
import math

from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
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
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure


class CombinedMeasureFactory:
    """
    Factory class created to wrap the methods that allow creation of
    the different types of `CombinedMeasureBase`.
    """

    @staticmethod
    def get_shsg_combined_measure(
        shsg_measure_collection: list[ShSgMeasure],
        sh_combination: ShCombinedMeasure,
        sg_combination: SgCombinedMeasure,
    ) -> ShSgCombinedMeasure | None:
        """
        Retrieves a new `ShSgCombinedMeasure` when the provided combinations
        can be matched with any of the items in the `ShSgMeasure` list.

        Args:
            shsg_measure_collection (list[ShSgMeasure]): List of `ShSgMeasure` candidates.
            sh_combination (ShCombinedMeasure): Sh combination.
            sg_combination (SgCombinedMeasure): Sg combination.

        Returns:
            ShSgCombinedMeasure | None: New combined measure or `None` when no match was found.
        """

        def check_sh_sg_measures_match(shsg_combination: ShSgCombinedMeasure) -> bool:
            def floats_are_equal_or_nan(left_float: float, right_float: float) -> bool:
                """
                Compares two floats for equality, when both are `float("nan")` then
                we considered them as equal.
                """
                if math.isnan(left_float) and math.isnan(right_float):
                    return True
                return left_float == right_float

            return (
                floats_are_equal_or_nan(
                    shsg_combination.dcrest, sh_combination.primary.dcrest
                )
                and floats_are_equal_or_nan(
                    shsg_combination.dberm, sg_combination.primary.dberm
                )
                and floats_are_equal_or_nan(
                    shsg_combination.l_stab_screen, sg_combination.primary.l_stab_screen
                )
                and shsg_combination.measure_type == sh_combination.primary.measure_type
                and shsg_combination.year == sh_combination.primary.year
            )

        _found_shsg_measures = list(
            filter(check_sh_sg_measures_match, shsg_measure_collection)
        )
        if not _found_shsg_measures:
            return None

        if len(_found_shsg_measures) > 1:
            logging.warning(
                "More than one `ShSgMeasure` found for combination of primary measure results (%s, %s). Using only the first one found.",
                sh_combination.primary.measure_result_id,
                sg_combination.primary.measure_result_id,
            )
        _shsg_measure = _found_shsg_measures[0]

        return ShSgCombinedMeasure(
            primary=_shsg_measure,
            sh_secondary=sh_combination.secondary,
            sg_secondary=sg_combination.secondary,
            mechanism_year_collection=_shsg_measure.mechanism_year_collection,
        )

    @staticmethod
    def from_input(
        primary: MeasureAsInputProtocol,
        secondary: MeasureAsInputProtocol | None,
        initial_assessment: MechanismPerYearProbabilityCollection,
        sequence_nr: int,
    ) -> CombinedMeasureBase:
        """
        Create a combined measure from input

        Args:
            primary (MeasureAsInputProtocol): The primary measure
            secondary (MeasureAsInputProtocol | None): The secondary measure
            initial_assessment (MechanismPerYearProbabilityCollection): The initial assessment
            sequence_nr (int): The sequence nr of the combination in the list of Sg- or Sh-combinations

        Returns:
            CombinedMeasureBase: The combined measure
        """
        _mech_year_coll = primary.mechanism_year_collection
        if secondary:
            _mech_year_coll = MechanismPerYearProbabilityCollection.combine(
                primary.mechanism_year_collection,
                secondary.mechanism_year_collection,
                initial_assessment,
            )

        _combined_measure_dict = dict(
            primary=primary,
            secondary=secondary,
            mechanism_year_collection=_mech_year_coll,
            sequence_nr=sequence_nr,
        )
        if isinstance(primary, ShMeasure):
            return ShCombinedMeasure(**_combined_measure_dict)
        elif isinstance(primary, SgMeasure):
            return SgCombinedMeasure(**_combined_measure_dict)
        else:
            raise NotImplementedError(
                f"It is not supported to combine measures of type {type(primary).__name__}."
            )
