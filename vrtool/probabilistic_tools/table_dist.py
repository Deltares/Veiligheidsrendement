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
    ) -> None:
        super(TableDist, self).__init__(1)

        if len(x) != len(p):
            raise ValueError("Input arrays have unequal lengths")

        if not extrap:
            if p[0] != 0.0 or p[-1] != 1.0:
                raise ValueError(
                    "Probability bounds are not equal to 0 and 1. Allow for extrapolation or change input"
                )

        if np.any(np.diff(x) < 0):
            raise ValueError("Values should be increasing")
        if np.any(np.diff(p) < 0):
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
        spline = InterpolatedUnivariateSpline(p, x, k=1)
        xgrid = spline(pgrid)
        if xgrid[0] > xgrid[-1]:
            self.x = np.flip(xgrid)
            self.p = np.flip(pgrid)
            self.p[0] = 0.0
        else:
            self.x = xgrid
            self.p = pgrid
            self.p[-1] = 1.0

    @staticmethod
    def _compute_decimation_height(h: np.ndarray, p: np.ndarray, n: int = 2):
        # computes the average decimation height for the lower parts of a distribution: h are water levels, p are exceedence probabilities. n is the number of 'decimations'
        hp = interp1d(p, h)
        h_low = hp(p[0])  # lower limit
        h_high = hp((p[0]) / (10 * n))
        return (h_high - h_low) / n

    def computeCDF(self, X: float) -> float:
        if X < self.x[0]:
            return 0.0
        elif X >= self.x[-1:]:
            return 1.0
        return float(np.interp(X, self.x, self.p))

    def getMean(self) -> float:
        high = np.min(np.argwhere(self.p > 0.53))
        low = np.min(np.argwhere(self.p > 0.47))
        index = low + (np.abs(0.5 - self.p[low:high])).argmin()
        mu = np.interp(
            0.5, self.p[index - 1 : index + 1], self.x[index - 1 : index + 1]
        )
        return [mu]

    def getRange(self) -> ot.Interval:
        return ot.Interval([self.x[0]], [float(self.x[-1:])], [True], [True])
