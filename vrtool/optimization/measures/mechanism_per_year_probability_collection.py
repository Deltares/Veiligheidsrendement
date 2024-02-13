from __future__ import annotations

from scipy.interpolate import interp1d

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class MechanismPerYearProbabilityCollection:
    probabilities: list[MechanismPerYear]

    def __init__(self, probabilities: list[MechanismPerYear]) -> None:
        self.probabilities = probabilities

    def filter(self, mechanism: MechanismEnum, year: int) -> float:
        """
        filter probabilities given the mechanism and year

        Args:
            mechanism (MechanismEnum): the mechanism to filter
            year (int): the year to filter

        Raises:
            ValueError: combination of mechanism and year not available

        Returns:
            float: the probability
        """
        for p in self.probabilities:
            if p.mechanism == mechanism and p.year == year:
                return p.probability
        raise ValueError("mechanism/year not found")

    def get_mechanisms(self) -> set[MechanismEnum]:
        """
        get the mechanisms used in _probabilities

        Returns:
            set[MechanismEnum]: a set with the mechanisms
        """
        return set(p.mechanism for p in self.probabilities)

    def get_years(self, mechanism: MechanismEnum) -> set[int]:
        """
        get the years in _probabilities for a given mechanism

        Args:
            mechanism (MechanismEnum): the mechanism to filter

        Returns:
            set[int]: a set with the years
        """
        return set(p.year for p in self.probabilities if p.mechanism == mechanism)

    @staticmethod
    def _combine_probs_for_mech(
        mechanism: MechanismEnum,
        primary: MechanismPerYearProbabilityCollection,
        secondary: MechanismPerYearProbabilityCollection,
    ) -> list[MechanismPerYear]:
        """
        helper routine for combine: combines for one mechanism.

        Args:
            mechanism (MechanismEnum): the mechanism
            secondary (MechanismPerYearProbabilityCollection): the second measure

        Returns:
            list[MechanismPerYear]: resulting list with probabilities
        """
        _nw_list = []
        for _mech_per_year in filter(
            lambda x: x.mechanism == mechanism, primary.probabilities
        ):
            _prob_second = secondary.filter(mechanism, _mech_per_year.year)
            _nwp = (
                _mech_per_year.probability
                + _prob_second
                - _mech_per_year.probability * _prob_second
            )
            _nw_list.append(MechanismPerYear(mechanism, _mech_per_year.year, _nwp))
        return _nw_list

    @classmethod
    def combine(
        cls,
        primary: MechanismPerYearProbabilityCollection,
        secondary: MechanismPerYearProbabilityCollection,
    ) -> MechanismPerYearProbabilityCollection:
        """
        Combines the probabilities in two collections.
        Years and mechanisms of both collections should match.

        Args:
            second (MechanismPerYearProbabilityCollection): the collection to combine with

        Raises:
            ValueError: can only combine if the overlying mechanisms are equal

        Returns:
            list[MechanismPerYear]: the combined collection
        """
        _mechanism1 = primary.get_mechanisms()
        _mechanism2 = secondary.get_mechanisms()
        if _mechanism1 != _mechanism2:  # TODO improve check
            raise ValueError("mechanisms not equal in combine")
        _nw_probabilities = []
        for m in _mechanism1:
            _years1 = primary.get_years(m)
            _years2 = secondary.get_years(m)
            if _years1 != _years2:  # TODO improve check
                raise ValueError("years not equal in combine")
            _nw_probabilities.extend(cls._combine_probs_for_mech(m, primary, secondary))
        return cls(_nw_probabilities)

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
        for p in self.probabilities:
            if p.mechanism == mechanism:
                _years.append(p.year)
                _betas.append(p.beta)

        _beta_interp = interp1d(_years, _betas, fill_value="extrapolate")(added_years)
        for i, _year in enumerate(added_years):
            if _year not in _years:
                _mech_per_yr = MechanismPerYear(
                    mechanism, _year, beta_to_pf(float(_beta_interp[i]))
                )
                self.probabilities.append(_mech_per_yr)

    def add_years(self, years: list[int]) -> None:
        """
        Extend probabilities with more years using interpolation.
        Interpolation is based on betas.
        This is done per mechanism. Duplication of years is avoided.

        Args:
            years (list[int]): years to add.
        """
        _mechanisms = self.get_mechanisms()
        for m in _mechanisms:
            self._add_year_mechanism(m, years)
