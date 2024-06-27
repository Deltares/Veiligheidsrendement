from vrtool.optimization.measures.combined_measures.combined_measure_protocol import (
    CombinedMeasureProtocol,
)
from vrtool.optimization.measures.combined_measures.sg_combined_measure import (
    SgCombinedMeasure,
)
from vrtool.optimization.measures.combined_measures.sh_combined_measure import (
    ShCombinedMeasure,
)
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


class CombinedMeasureFactory:
    @staticmethod
    def from_input(
        primary: MeasureAsInputProtocol,
        secondary: MeasureAsInputProtocol | None,
        initial_assessment: MechanismPerYearProbabilityCollection,
        sequence_nr: int,
    ) -> CombinedMeasureProtocol:
        """
        Create a combined measure from input

        Args:
            primary (MeasureAsInputProtocol): The primary measure
            secondary (MeasureAsInputProtocol | None): The secondary measure
            initial_assessment (MechanismPerYearProbabilityCollection): The initial assessment
            sequence_nr (int): The sequence nr of the combination in the list of Sg- or Sh-combinations

        Returns:
            CombinedMeasure: The combined measure
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
                f"It is not supported to combine measures of type {type(primary)}"
            )
