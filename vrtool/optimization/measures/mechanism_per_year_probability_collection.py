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

    def _get_years(self, mechanism: MechanismEnum) -> set[int]:
        return set(p.year for p in self._probabilities if p.mechanism == mechanism)

    def _combine_years(self, mechanism: MechanismEnum, secondary_list: list[MechanismPerYear], nw_list: list[MechanismPerYear]):
        """
        helper routine for combin: combines for one mechanism.

        Args:
            mechanism (MechanismEnum): the mechanism
            secondary_list (list[MechanismPerYear]): list with probabilities from second measure
            nw_list (list[MechanismPerYear]): resulting list with probabilities (inout)
        """
        for p in secondary_list:
            if p.mechanism == mechanism:
                _prob_second = self.filter(mechanism, p.year)
                _nwp = p.probability + _prob_second - p.probability * _prob_second
                nw_list.append(MechanismPerYear(mechanism, p.year, _nwp))

    def combine(self, second: MechanismPerYearProbabilityCollection) -> list[MechanismPerYear]:
        """
        combines the probabilities in two collections

        Args:
            second (MechanismPerYearProbabilityCollection): the collection to combine with

        Raises:
            ValueError: can only combine if the overlying mechanisms are equal

        Returns:
            list[MechanismPerYear]: the combined collection
        """        
        _mechanism1 = self._get_mechanisms()
        _mechanism2 = second._get_mechanisms()
        if ( not (_mechanism1 == _mechanism2)):
            raise ValueError("mechanisms not equal in combine")
        _nw_probabilities = []
        for m in _mechanism1:
            _years1 = self._get_years(m)
            _years2 = second._get_years(m)
            if (_years1 != _years2):
                raise ValueError("years not equal in combine")
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
