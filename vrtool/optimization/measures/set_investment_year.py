from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


class SetInvestmentYear:
    def update_measurelist_with_investment_year(
        self,
        measures: list[MeasureAsInputProtocol],
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

        self._add_investment_years(measures, initial)

        for measure in measures:
            if measure.year > 0:
                self._update_measure(measure, initial)

    def _add_investment_years(
        self,
        measures: list[MeasureAsInputProtocol],
        initial: MechanismPerYearProbabilityCollection,
    ) -> None:
        _investment_years = set()
        for measure in measures:
            if measure.year > 0:
                _investment_years.add(measure.year)
                _investment_years.add(measure.year + 1)

        if len(_investment_years) > 0:
            _years = list(_investment_years)
            initial.add_years(_years)
            for measure in measures:
                measure.mechanism_year_collection.add_years(_years)

    def _update_measure(
        self,
        measure: MeasureAsInputProtocol,
        initial: MechanismPerYearProbabilityCollection,
    ) -> None:
        _investment_year = measure.year
        measure.mechanism_year_collection.replace_values(initial, _investment_year)
