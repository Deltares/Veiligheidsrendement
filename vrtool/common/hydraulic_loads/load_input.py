from dataclasses import dataclass, field

import numpy as np
import openturns as ot
from scipy.interpolate import interp1d

from vrtool.probabilistic_tools.table_dist import TableDist


@dataclass
class LoadInput:
    # class to store load data
    distribution: dict[int, ot.Distribution] = field(default_factory=dict)

    def set_distribution(
        self,
        year: int,
        wls: np.ndarray,
        p_nexc: np.ndarray,
        gridpoints: int,
    ):
        self.distribution[year] = ot.Distribution(
            TableDist(wls, p_nexc, extrap=True, isload=True, gridpoints=gridpoints)
        )

    def compute_h(self, year: int, p: float) -> float:
        def compute(year: int, p: float) -> float:
            return self.distribution[year].computeQuantile(p)[0]

        if year in self.distribution.keys():
            return compute(year, p)
        else:
            _years = list(self.distribution.keys())
            _wls = [compute(_year, p) for _year in _years]
            return interp1d(_years, _wls, fill_value="extrapolate")(year)
