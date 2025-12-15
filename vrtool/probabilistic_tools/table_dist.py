from typing import override

import numpy as np
import openturns as ot
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d


class TableDist(ot.PythonDistribution):
    def __init__(
        self,
        x: np.ndarray,
        p: np.ndarray,
        extrap: bool = False,
        isload: bool = False,
        gridpoints: int = 2000,
    ):
        super(TableDist, self).__init__(1)
        # Check the input
        if len(x) != len(p):
            raise ValueError("Input arrays have unequal lengths")
        if not extrap:
            if p[0] != 1 or p[-1:] != 0:
                raise ValueError(
                    "Probability bounds are not equal to 0 and 1. Allow for extrapolation or change input"
                )
        for i in range(1, len(x)):
            if x[i - 1] > x[i]:
                raise ValueError("Values should be increasing")
            if p[i - 1] > p[i]:
                raise ValueError("Non-exceedance probabilities should be increasing")

        # Define the distribution
        if isload:
            pgrid = 1 - np.logspace(0, -8, gridpoints)
            # we add a zero point to prevent excessive extrapolation. We do this based on the decimation height from the inserted points.
            d10 = self._compute_decimation_height(x, 1 - p)
            p_low = 1 - p[0]
            # determine water level with 100\% chance of occuring in a year
            p = np.concatenate(([0.0], p))
            x_low = x[0] - (1 / p_low) * (d10 / 10)
            x = np.concatenate(([x_low], x))
        else:
            pgrid = np.logspace(0, -8, gridpoints)

        # do inter/extrapolation
        s = InterpolatedUnivariateSpline(p, x, k=1)
        xgrid = s(pgrid)
        if xgrid[0] - xgrid[-1:] > 0:
            self.x = np.flip(xgrid, 0)
            self.xp = np.flip(pgrid, 0)
            self.xp[0] = 0.0

        else:
            self.x = xgrid
            self.xp = pgrid
            self.xp[-1:] = 1.0

    @staticmethod
    def _compute_decimation_height(h, p, n=2):
        # computes the average decimation height for the lower parts of a distribution: h are water levels, p are exceedence probabilities. n is the number of 'decimations'
        hp = interp1d(p, h)
        h_low = hp(p[0])  # lower limit
        h_high = hp((p[0]) / (10 * n))
        return (h_high - h_low) / n

    @override
    def computeCDF(self, X):
        if X < self.x[0]:
            return 0.0
        elif X >= self.x[-1:]:
            return 1.0
        else:
            X = X[0]
            idx_up = np.argmax(self.x > X)
            xx = self.x[idx_up - 1 : idx_up + 1]
            pp = self.xp[idx_up - 1 : idx_up + 1]
            dp = pp[1] - pp[0]
            dx = xx[1] - xx[0]
            p = pp[0] + dp * ((X - xx[0]) / dx)

            return p

    def getMean(self):
        high = np.min(np.argwhere(self.xp > 0.53))
        low = np.min(np.argwhere(self.xp > 0.47))
        index = low + (np.abs(0.5 - self.xp[low:high])).argmin()
        mu = np.interp(
            0.5, self.xp[index - 1 : index + 1], self.x[index - 1 : index + 1]
        )
        return [mu]

    def getRange(self):
        return ot.Interval([self.x[0]], [float(self.x[-1:])], [True], [True])
