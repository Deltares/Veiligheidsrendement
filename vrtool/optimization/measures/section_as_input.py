from dataclasses import dataclass, field
from typing import Optional

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
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
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure


@dataclass
class SectionAsInput:
    section_name: str
    traject_name: str
    flood_damage: float
    section_length: float
    measures: list[MeasureAsInputProtocol]
    initial_assessment: MechanismPerYearProbabilityCollection = field(
        default_factory=lambda: MechanismPerYearProbabilityCollection([])
    )
    combined_measures: list[CombinedMeasureBase] = field(
        default_factory=list[CombinedMeasureBase]
    )  # TODO do we need this in SectionAsInput or can it be volatile?
    aggregated_measure_combinations: Optional[
        list[AggregatedMeasureCombination]
    ] = field(default_factory=list[AggregatedMeasureCombination])

    def get_measures_by_class(
        self,
        measure_class: type[MeasureAsInputProtocol],
    ) -> list[MeasureAsInputProtocol]:
        """
        Get the measures for a section based on class of measure (Sg/Sh).

        Args:
            measure_class (type[MeasureAsInputProtocol]): Class of measure.

        Returns:
            list[MeasureAsInputProtocol]: Measures of the class.
        """
        return list(filter(lambda x: isinstance(x, measure_class), self.measures))

    def get_aggregated_combinations(
        self, sh_sequence_nr: int, sg_sequence_nr: int
    ) -> list[AggregatedMeasureCombination]:
        """
        Gets the list of aggregated combinations that match the given sequence numbers.

        Args:
            sh_sequence_nr (int): ShMeasure sequence number.
            sg_sequence_nr (int): SgMeasure sequence number.

        Returns:
            list[AggregatedMeasureCombination]: Aggregated measure combination collection matching the sequence numbers.
        """
        return [
            _amc
            for _amc in self.aggregated_measure_combinations
            if _amc.sh_combination.sequence_nr == sh_sequence_nr
            and _amc.sg_combination.sequence_nr == sg_sequence_nr
        ]

    @property
    def sh_measures(self) -> list[ShMeasure]:
        return self.get_measures_by_class(ShMeasure)

    @property
    def sg_measures(self) -> list[SgMeasure]:
        return self.get_measures_by_class(SgMeasure)

    @property
    def sh_sg_measures(self) -> list[ShSgMeasure]:
        return self.get_measures_by_class(ShSgMeasure)

    def _get_combinations_by_type(
        self, measure_type: type[CombinedMeasureBase]
    ) -> list[CombinedMeasureBase]:
        return list(
            filter(lambda x: isinstance(x, measure_type), self.combined_measures)
        )

    @property
    def sh_combinations(self) -> list[ShCombinedMeasure]:
        """
        Gets the Sh combinations for this `SectionAsInput`.

        Returns:
            list[ShCombinedMeasure]: Sh combined measures of the class.
        """
        return self._get_combinations_by_type(ShCombinedMeasure)

    @property
    def sg_combinations(self) -> list[SgCombinedMeasure]:
        """
        Gets the Sg combinations for this `SectionAsInput`.

        Returns:
            list[SgCombinedMeasure]: Sg combined measures of the class.
        """
        return self._get_combinations_by_type(SgCombinedMeasure)

    def _get_sample_years(self) -> set[int]:
        """
        Gets a list of the available years given the assumptions:
        - We have measure(s).
        - The first measure has mechanism(s).
        - The first mechanism has the same year(s) as the rest of mechanism(s) for all measures.

        Returns:
            set[int]: Unique collection of years
        """
        if not self.measures:
            return {0}
        # Get the max year for all measures for a random mechanism
        _sample_measure = self.measures[0]
        _sample_mechanism = _sample_measure.get_allowed_mechanisms()[0]
        return _sample_measure.mechanism_year_collection.get_years(_sample_mechanism)

    @property
    def max_year(self) -> int:
        """
        The maximum year for the section.
        Assumption: All measures and mechanisms have the same years.

        Returns:
            int: The maximum year
        """
        return max(self._get_sample_years())

    @property
    def min_year(self) -> int:
        """
        The minimum year for the section.

        Returns:
            int: The minimum year
        """
        return min(self._get_sample_years())

    @property
    def mechanisms(self) -> set[MechanismEnum]:
        """
        All mechanisms for the section

        Returns:
            set[MechanismEnum]: Set of mechanisms
        """
        if not self.measures:
            return set()
        return set(
            mech
            for meas in self.measures
            for mech in meas.mechanism_year_collection.get_mechanisms()
        )

    def update_measurelist_with_investment_year(self) -> None:
        """
        Update the probabilities for all measures.
        Measures with investment year > 0 get values from the zero measure.
        Other measures only get more years in mechanism_year_collection,
        to keep the number of years equal in a section.
        """

        _initial = self.initial_assessment

        _investment_years = self._get_investment_years()

        if len(_investment_years) == 0:
            return

        _initial.add_years(_investment_years)
        for measure in self.measures:
            measure.mechanism_year_collection.add_years(_investment_years)
            if measure.year > 0:
                measure.mechanism_year_collection.replace_values(_initial, measure.year)

    def _get_investment_years(self) -> list[int]:
        _investment_years = set()
        for measure in self.measures:
            if measure.year > 0:
                _investment_years.add(measure.year - 1)
                _investment_years.add(measure.year)
        return list(_investment_years)
