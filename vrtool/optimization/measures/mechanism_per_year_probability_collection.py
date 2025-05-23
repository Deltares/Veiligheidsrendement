from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class MechanismPerYearProbabilityCollection:
    probabilities: list[MechanismPerYear]

    def __init__(self, probabilities: list[MechanismPerYear]) -> None:
        self.probabilities = probabilities

    def get_probability(self, mechanism: MechanismEnum, year: int) -> float:
        """
        get the probability for a given mechanism and year

        Args:
            mechanism (MechanismEnum): the mechanism to filter
            year (int): the year to filter

        Raises:
            ValueError: combination of mechanism and year not available

        Returns:
            float: the probability
        """
        return next(
            (
                p.probability
                for p in self.probabilities
                if p.mechanism == mechanism and p.year == year
            ),
        )

    def get_beta(self, mechanism: MechanismEnum, year: int) -> float:
        """
        get the beta for a given mechanism and year

        Args:
            mechanism (MechanismEnum): the mechanism to filter
            year (int): the year to filter

        Returns:
            float: the beta
        """
        return next(
            (
                p.beta
                for p in self.probabilities
                if p.mechanism == mechanism and p.year == year
            ),
        )

    def get_probabilities(
        self, mechanism: MechanismEnum, years: list[int]
    ) -> list[float]:
        """
        Get the probabilites for a given mechanism and years.
        Interpolation is used to get the probabilities for the years that are not part of the collection.

        Args:
            mechanism (MechanismEnum): Mechanism
            years (list[int]): List of yearss

        Returns:
            list[float]: List of probabilities
        """
        _betas = self.get_betas(mechanism, years)
        if not _betas:
            return []
        return beta_to_pf(_betas)

    def get_betas(self, mechanism: MechanismEnum, years: list[int]) -> list[float]:
        """
        Get the betas for a given mechanism and years.
        Interpolation is used to get the betas for the years that are not part of the collection.

        Args:
            mechanism (MechanismEnum): Mechanism
            years (list[int]): List of years

        Returns:
            list[float]: List of betas
        """
        _years = list(self.get_years(mechanism))
        if not _years:
            return []
        _betas = list(map(lambda x: self.get_beta(mechanism, x), _years))
        return interp1d(_years, _betas, fill_value="extrapolate")(years).tolist()

    def get_mechanisms(self) -> set[MechanismEnum]:
        """
        get the mechanisms used in probabilities

        Returns:
            set[MechanismEnum]: a set with the mechanisms
        """
        return set(p.mechanism for p in self.probabilities)

    def get_years(self, mechanism: MechanismEnum) -> set[int]:
        """
        get the years in probabilities for a given mechanism

        Args:
            mechanism (MechanismEnum): the mechanism to filter

        Returns:
            set[int]: a set with the years
        """
        return set(p.year for p in self.probabilities if p.mechanism == mechanism)

    def get_section_probability(self, year: int) -> float:
        """
        get the section probability for a given year for all mechanisms present

        Args:
            year(int): the year for which to get the combined section probability

        Returns:
            float: the combined section probability
        """
        list_of_probabilities = [
            self.get_probability(mechanism, year) for mechanism in self.get_mechanisms()
        ]
        return 1 - np.subtract(1, list_of_probabilities).prod()

    @staticmethod
    def _combine_probs_for_mech(
        mechanism: MechanismEnum,
        primary: MechanismPerYearProbabilityCollection,
        secondary: MechanismPerYearProbabilityCollection,
        initial: MechanismPerYearProbabilityCollection,
    ) -> list[MechanismPerYear]:
        """
        Helper routine for combine: combines for one mechanism.
        If probabilities of primary and initial are equal, only secondary is used.
        If probabilities of secondary and initial are equal, only primary is used.
        In all other cases the primary and secondary are combined, using 1 - (1-primary) * (1-secondary).

        Args:
            mechanism (MechanismEnum): the mechanism
            secondary (MechanismPerYearProbabilityCollection): the second measure
            initial (MechanismPerYearProbabilityCollection): the initial assessment collection

        Returns:
            list[MechanismPerYear]: resulting list with probabilities
        """
        _nw_list = []
        for _mech_per_year in filter(
            lambda x: x.mechanism == mechanism, primary.probabilities
        ):
            _prob_second = secondary.get_probability(mechanism, _mech_per_year.year)
            _prob_initial = initial.get_probability(mechanism, _mech_per_year.year)

            if _mech_per_year.probability == _prob_initial:
                _nwp = _prob_second
            elif _prob_second == _prob_initial:
                _nwp = _mech_per_year.probability
            else:
                if mechanism != MechanismEnum.PIPING:
                    raise ValueError("This should not happen")
                # TODO: make exact formula, now it is an approximation that gives very small differences
                _ratio_improved = (
                    _prob_initial / _prob_second
                )  # ratio of improvement of secondary over initial (so no measure vs only VZG)
                _nwp = _mech_per_year.probability / _ratio_improved
            _nw_list.append(MechanismPerYear(mechanism, _mech_per_year.year, _nwp))
        return _nw_list

    @classmethod
    def combine(
        cls,
        primary: MechanismPerYearProbabilityCollection,
        secondary: MechanismPerYearProbabilityCollection,
        initial: MechanismPerYearProbabilityCollection,
    ) -> MechanismPerYearProbabilityCollection:
        """
        Combines the probabilities in two collections.
        Years and mechanisms of both collections should match.

        Args:
            primary (MechanismPerYearProbabilityCollection): the primary collection
            second (MechanismPerYearProbabilityCollection): the collection to combine with
            initial (MechanismPerYearProbabilityCollection): the initial assessment collection

        Raises:
            ValueError: can only combine if the overlying mechanisms are equal

        Returns:
            MechanismPerYearProbabilityCollection: the combined collection
        """
        _mechanism_prim = primary.get_mechanisms()
        _mechanism_sec = secondary.get_mechanisms()
        if _mechanism_prim != _mechanism_sec:
            raise ValueError("mechanisms not equal in combine")
        _nw_probabilities = []
        for m in _mechanism_prim:
            _years_prim = primary.get_years(m)
            _years_sec = secondary.get_years(m)
            if _years_prim != _years_sec:
                raise ValueError("years not equal in combine")
            _nw_probabilities.extend(
                cls._combine_probs_for_mech(m, primary, secondary, initial)
            )
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

        _beta_interp = interp1d(_years, _betas, fill_value="extrapolate")(
            added_years
        ).tolist()
        if not _beta_interp:
            return
        for i, _year in enumerate(added_years):
            if _year not in _years:
                _mech_per_year = MechanismPerYear(
                    mechanism, _year, beta_to_pf(float(_beta_interp[i]))
                )
                self.probabilities.append(_mech_per_year)

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

    def _replace(self, mechanism: MechanismEnum, year: int, prob: float) -> None:
        for p in self.probabilities:
            if p.mechanism == mechanism and p.year == year:
                p.probability = prob
                return
        raise ValueError("mechanism and year not found in replace")

    def _replace_values_mechanism(
        self,
        mechanism: MechanismEnum,
        year_zero_values: MechanismPerYearProbabilityCollection,
        investment_year: int,
    ) -> None:
        _years = self.get_years(mechanism)
        for yr in _years:
            if yr < investment_year:
                _nwprob = year_zero_values.get_probability(mechanism, yr)
                self._replace(mechanism, yr, _nwprob)

    def replace_values(
        self,
        initial: MechanismPerYearProbabilityCollection,
        investment_year: int,
    ):
        """
        Replace probabilities for years before investment_year with values
        from the measure with investment_year = 0.
        Assumes that these years are available.

        Args:
            initial (MechanismPerYearProbabilityCollection): the initial probabilities
            investment_year (int): the investment year

        Raises:
            ValueError: mismatch in mechanisms
        """
        _mechanism1 = self.get_mechanisms()
        _mechanism2 = initial.get_mechanisms()
        if not _mechanism1.issubset(_mechanism2):
            raise ValueError("mechanism not found in replace_values")
        for m in _mechanism1:
            self._replace_values_mechanism(m, initial, investment_year)
