from __future__ import annotations

from scipy.interpolate import interp1d

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class MechanismPerYearProbabilityCollection:
    _probabilities: list[MechanismPerYear]

    def __init__(self, probabilities: list[MechanismPerYear]) -> None:
        self._probabilities = probabilities

    def filter(self, mechanism: MechanismEnum, year: int) -> float:
        for p in self._probabilities:
            if p.mechanism == mechanism and p.year == year:
                return p.probability
        raise ValueError("mechanism/year not found")

    def _get_mechanisms(self) -> set[MechanismEnum]:
        return set(p.mechanism for p in self._probabilities)

    def _combine_years(self, mechanism: MechanismEnum, secondary_list: list[MechanismPerYear], nw_list: list[MechanismPerYear]):
        for p in secondary_list:
            if p.mechanism == mechanism:
                y = self.filter(mechanism, p.year)
                nwp = 1 - (1 - p.probability) * (1 - y)
                nw_list.append(MechanismPerYear(mechanism, p.year, nwp))

    def combine(self, second: MechanismPerYearProbabilityCollection):
        _mechanism1 = self._get_mechanisms()
        _mechanism2 = second._get_mechanisms()
        if ( not (_mechanism1 == _mechanism2)):
            raise ValueError("mechanisms not equal in combine")
        _nw_probabilities = []
        for m in _mechanism1:
            self._combine_years(m, second._probabilities, _nw_probabilities)
        return _nw_probabilities

    def _add_year_mechanism(
        self, mechanism: MechanismEnum, added_years: list[int]
    ) -> None:
        """
        helper method for add_years: extend probabilities with more years using interpolation.
        Interpolation of betas and avoid duplication of years.

        Args:
            mechanism (MechanismEnum): mechanism that is extended
            added_years (list[int]): years to add
        """
        _years = []
        _betas = []
        for p in self._probabilities:
            if p.mechanism == mechanism:
                _years.append(p.year)
                _betas.append(p.beta)

        _beta_interp = interp1d(_years, _betas, fill_value=("extrapolate"))(added_years)
        for i, _year in enumerate(added_years):
            if _year not in _years:
                _mech_per_year = MechanismPerYear(
                    mechanism, _year, beta_to_pf(float(_beta_interp[i]))
                )
                self._probabilities.append(_mech_per_year)

    def add_years(self, years: list[int]) -> None:
        """
        Extend probabilities with more years using interpolation.
        Interpolation is based on betas.
        This is done per mechanism. Duplication of years is avoided.

        Args:
            years (list[int]): years to add.
        """
        _mechanisms = self._get_mechanisms()
        for m in _mechanisms:
            self._add_year_mechanism(m, years)
