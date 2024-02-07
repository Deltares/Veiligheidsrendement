from scipy.interpolate import interp1d

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
    
    def _get_mechanisms(self) -> {MechanismEnum}:
        myset = set()
        for p in self._probabilities:
            myset.add(p.mechanism)
        return myset
    
    def _add_year_mechanism(self, mechanism: MechanismEnum, added_years: list[int]) -> None:
        years = []
        probs = []
        for p in self._probabilities:
            if p.mechanism == mechanism:
                years.append(p.year)
                probs.append(p.probability)

        prob_interp = interp1d(years, probs, fill_value=("extrapolate"))(added_years)
        for i in range(len(added_years)):
            mechPerYr = MechanismPerYear(mechanism, added_years[i], float(prob_interp[i]))
            self._probabilities.append(mechPerYr)

    def add_years(self, years: list[int]) -> None:
        mechanisms = self._get_mechanisms()
        for m in mechanisms:
            self._add_year_mechanism(m, years)

