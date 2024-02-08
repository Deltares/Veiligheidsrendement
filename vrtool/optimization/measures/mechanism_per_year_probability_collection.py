from scipy.interpolate import interp1d
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear


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

    def _add_year_mechanism(self, mechanism: MechanismEnum, added_years: list[int]) -> None:
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
            if (not (_year in _years)):
                _mechPerYr = MechanismPerYear(mechanism, _year, beta_to_pf(float(_beta_interp[i])))
                self._probabilities.append(_mechPerYr)

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

