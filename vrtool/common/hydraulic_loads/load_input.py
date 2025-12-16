from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

from vrtool.probabilistic_tools.table_dist import TableDist


@dataclass
class LoadInput:
    # class to store load data
    distribution: dict[int, TableDist] = field(default_factory=dict)

    def set_distribution(
        self,
        year: int,
        wls: np.ndarray,
        p_nexc: np.ndarray,
        gridpoints: int,
    ):
        self.distribution[year] = TableDist(
            wls, p_nexc, extrap=True, isload=True, gridpoints=gridpoints
        )

    def compute_h(self, year: int, p: float) -> float:
        def compute(year: int) -> float:
            return self.distribution[year].computeQuantile(p)

        if year in self.distribution.keys():
            return compute(year)
        else:
            _years = list(self.distribution.keys())
            _values = [compute(_year) for _year in _years]
            return float(interp1d(_years, _values, fill_value="extrapolate")(year))
