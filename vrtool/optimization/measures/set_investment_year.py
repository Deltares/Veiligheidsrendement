from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)


class SetInvestmentYear:
    def update_measurelist_with_investment_year(
        self, measures: list[MeasureAsInputProtocol]
    ) -> None:
        """
        update the probabilities for measures with investment year > 0.

        Args:
            measures (list[MeasureAsInputProtocol]): list with all measures
        """
        for measure in measures:
            if measure.year > 0:
                _measure_zero = self._find_measure_with_year_zero(measure, measures)
                self._update_measure(measure, _measure_zero)

    def _find_measure_with_year_zero(
        self, measure: MeasureAsInputProtocol, measures: list[MeasureAsInputProtocol]
    ) -> MeasureAsInputProtocol:
        for m in measures:
            if m.year == 0:
                if m.measure_type == measure.measure_type:
                    if m.equals_except_year(measure):
                        return m
        raise ValueError("equal measure for year=0 not found")

    def _update_measure(
        self, measure: MeasureAsInputProtocol, measure_zero: MeasureAsInputProtocol
    ) -> None:
        _investment_year = measure.year
        measure.mechanism_year_collection.add_years([_investment_year, _investment_year + 1])
        measure_zero.mechanism_year_collection.add_years([_investment_year])
        measure.mechanism_year_collection.replace_values(
            measure_zero.mechanism_year_collection, _investment_year
        )
