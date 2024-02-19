from dataclasses import dataclass, field
from typing import Optional
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum

from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


@dataclass
class SectionAsInput:
    section_name: str
    traject_name: str
    measures: list[MeasureAsInputProtocol]
    combined_measures: list[CombinedMeasure] = field(
        default_factory=list[CombinedMeasure]
    )  # TODO do we need this in SectionAsInput or can it be volatile?
    aggregated_measure_combinations: Optional[
        list[AggregatedMeasureCombination]
    ] = field(default_factory=list[AggregatedMeasureCombination])

    @property
    def initial_assessment(self) -> MechanismPerYearProbabilityCollection:
        _zero_sg = next(
            m.mechanism_year_collection
            for m in self.sg_measures
            if m.measure_type == MeasureTypeEnum.SOIL_REINFORCEMENT
            and m.year == 0
            and m.dberm == 0
            and m.dcrest == 0.0
        )
        _zero_sh = next(
            m.mechanism_year_collection
            for m in self.sh_measures
            if m.measure_type == MeasureTypeEnum.SOIL_REINFORCEMENT
            and m.year == 0
            and m.dcrest == 0.0
        )
        for p in _zero_sh.probabilities:
            _zero_sg.probabilities.append(p)

        return _zero_sg

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

    @property
    def sh_measures(self) -> list[MeasureAsInputProtocol]:
        return self.get_measures_by_class(ShMeasure)

    @property
    def sg_measures(self) -> list[MeasureAsInputProtocol]:
        return self.get_measures_by_class(SgMeasure)

    def get_combinations_by_class(
        self, measure_class: type[MeasureAsInputProtocol]
    ) -> list[CombinedMeasure]:
        """
        Get the combinations of measures for a section
        based on the class of measure of the primary measure (Sg/Sh).

        Args:
            measure_class (type[MeasureAsInputProtocol]): Class of measure.

        Returns:
            list[CombinedMeasure]: Combined measures of the class.
        """
        return list(
            filter(
                lambda x: isinstance(x.primary, measure_class), self.combined_measures
            )
        )

    @property
    def sh_combinations(self) -> list[CombinedMeasure]:
        return self.get_combinations_by_class(ShMeasure)

    @property
    def sg_combinations(self) -> list[CombinedMeasure]:
        return self.get_combinations_by_class(SgMeasure)

    def update_measurelist_with_investment_year(
        self,
        initial: MechanismPerYearProbabilityCollection,
    ) -> None:
        """
        Update the probabilities for all measures.
        Measures with investment year > 0 get values from the zero measure.
        Other measure only get more years in mechanism_year_collection,
        to keep the number of years equal in a section.

        Args:
            measures (list[MeasureAsInputProtocol]): list with all measures
            initial (MechanismPerYearProbabilityCollection): initial probabilities
        """

        _investment_years = self._get_investment_years()

        if len(_investment_years) == 0:
            return

        initial.add_years(_investment_years)
        for measure in self.measures:
            measure.mechanism_year_collection.add_years(_investment_years)

        for measure in self.measures:
            if measure.year > 0:
                measure.mechanism_year_collection.replace_values(initial, measure.year)

    def _get_investment_years(self) -> list[int]:
        _investment_years = set()
        for measure in self.measures:
            if measure.year > 0:
                _investment_years.add(measure.year)
                _investment_years.add(measure.year + 1)
        return list(_investment_years)
