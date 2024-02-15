from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)


class SetInvestmentYear:
    def update_measurelist_with_investment_year(
        self, measures: list[MeasureAsInputProtocol]
    ) -> None:
        """
        Update the probabilities for measures with investment year > 0.

        Raises:
            ValueError: zero measure not found
        Args:
            measures (list[MeasureAsInputProtocol]): list with all measures
        """
        for measure in measures:
            if measure.year > 0:
                _measure_zero = self._find_zero_measure(measure, measures)
                if _measure_zero:
                    self._update_measure(measure, _measure_zero)
                else:
                    _name = measure.measure_type.name
                    raise ValueError("zero measure not found for this type: " + _name)

    def _find_zero_measure(
        self, measure: MeasureAsInputProtocol, measures: list[MeasureAsInputProtocol] | None
    ) -> MeasureAsInputProtocol:
        for m in measures:
            if m.measure_type == measure.measure_type:
                if m.is_zero_measure():
                    return m
        return None

    def _update_measure(
        self, measure: MeasureAsInputProtocol, measure_zero: MeasureAsInputProtocol
    ) -> None:
        _investment_year = measure.year
        measure.mechanism_year_collection.add_years([_investment_year, _investment_year + 1])
        measure_zero.mechanism_year_collection.add_years([_investment_year])
        measure.mechanism_year_collection.replace_values(
            measure_zero.mechanism_year_collection, _investment_year
        )
