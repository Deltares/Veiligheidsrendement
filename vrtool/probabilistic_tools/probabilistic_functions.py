import logging

import numpy as np
import openturns as ot
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.stats import norm


def compute_decimation_height(h, p, n=2):
    # computes the average decimation height for the lower parts of a distribution: h are water levels, p are exceedence probabilities. n is the number of 'decimations'
    hp = interp1d(p, h)
    h_low = hp(p[0])  # lower limit
    h_high = hp((p[0]) / (10 * n))
    return (h_high - h_low) / n


class TableDist(ot.PythonDistribution):
    def __init__(self, x=[0, 1], p=[1, 0], extrap=False, isload=False, gridpoints=2000):
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
        pp1 = 1
        pp0 = 0
        if isload:
            pgrid = 1 - np.logspace(0, -8, gridpoints)
            # we add a zero point to prevent excessive extrapolation. We do this based on the decimation height from the inserted points.
            d10 = compute_decimation_height(x, 1 - p)
            p_low = 1 - p[0]
            # determine water level with 100\% chance of occuring in a year
            p = np.concatenate(([0.0], p))
            x_low = x[0] - (1 / p_low) * (d10 / 10)
            x = np.concatenate(([x_low], x))
        else:
            pgrid = np.logspace(0, -8, gridpoints)
            # pgrid = 1-np.logspace(0,-16,500)

        # spline order: 1 linear, 2 quadratic, 3 cubic ...
        order = 1
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

    def getParameterDescription(self):
        descr1 = []
        descr2 = []
        for i in range(0, len(self.x)):
            descr1.append("x_" + str(i))
            descr2.append("xp_" + str(i))
        return descr1 + descr2

    def getParameter(self):
        x = []
        xp = []
        for i in range(0, len(self.x)):
            x.append(self.x[i])
            xp.append(self.xp[i])
        return ot.Point(x + xp)

    def computeCDF(self, X):
        if X < self.x[0]:
            return 0.0
        elif X >= self.x[-1:]:
            return 1.0
        else:
            # find first value that is larger:
            # Option 1, seems to be slightly slower:
            # idx_up = min(np.argwhere(self.x>X))
            # xx = self.x[int(idx_up)-1:int(idx_up)+1]
            # pp = self.xp[int(idx_up)-1:int(idx_up)+1]
            # f = interp1d(xx,pp)
            # p = f(X)
            X = X[0]

            # idx_up = np.min(np.argwhere(self.x > X))
            idx_up = np.argmax(self.x > X)
            xx = self.x[idx_up - 1 : idx_up + 1]
            pp = self.xp[idx_up - 1 : idx_up + 1]
            dp = pp[1] - pp[0]
            dx = xx[1] - xx[0]
            p = pp[0] + dp * ((X - xx[0]) / dx)

            return p

    def computeQuantile_alternative(self, p, tail=False):
        if tail:  # if input p is to be interpreted as exceedence probability
            p = 1 - p
        # Linearly interpolate between two values

        # idx_up = np.min(np.argwhere(self.x > X))
        # find index above
        idx_up = np.argmax(self.xp > p)

        xx = self.x[idx_up - 1 : idx_up + 1]
        pp = self.xp[idx_up - 1 : idx_up + 1]
        dp = pp[1] - pp[0]
        dx = xx[1] - xx[0]
        x = xx[0] + dx * ((p - pp[0]) / dp)
        return x

    def getMean(self):
        high = np.min(np.argwhere(self.xp > 0.53))
        low = np.min(np.argwhere(self.xp > 0.47))
        # high = np.min(np.argwhere(self.xp > 0.50))
        # low = high-1
        index = low + (np.abs(0.5 - self.xp[low:high])).argmin()
        mu = np.interp(
            0.5, self.xp[index - 1 : index + 1], self.x[index - 1 : index + 1]
        )
        return [mu]

    def getRange(self):
        return ot.Interval([self.x[0]], [float(self.x[-1:])], [True], [True])

    def getRealization(self):
        X = []
        p = ot.RandomGenerator.Generate()
        idx_up = min(np.argwhere(self.xp > p))  # CHECK
        pp = self.xp[int(idx_up) - 1 : int(idx_up) + 1]
        xx = self.x[int(idx_up) - 1 : int(idx_up) + 1]
        f = interp1d(pp, xx)
        X = float(f(p))
        return ot.Point(1, X)

        # sample = h.getSample(50000)
        # from openturns.viewer import View
        # graph = ot.VisualTest_DrawEmpiricalCDF(sample)
        # orig = ot.Curve(wls, p_nexc)
        # graph.add(orig)
        # View(graph).show()

    def getSample(self, size):
        X = []
        for i in range(size):
            X.append(self.getRealization())
        return X


def add_load_char_vals(
    input, t_0: int, load, p_h: float, p_dh: float, year: float
) -> dict:
    # TODO this function should be moved elsewhere
    # input = list of all strength variables

    if load != None:
        if isinstance(load.distribution, dict):
            if str(np.int32(year + t_0)) in list(load.distribution.keys()):
                h_norm = np.array(
                    load.distribution[str(np.int32(year + t_0))].computeQuantile(
                        1 - p_h
                    )
                )[0]
            else:
                # for each year, compute WL
                years = [np.int32(i) for i in list(load.distribution.keys())]
                wls = []
                for _dist_year in years:
                    wls.append(
                        load.distribution[_dist_year].computeQuantile(1 - p_h)[0]
                    )
                h_norm = interp1d(years, wls, fill_value="extrapolate")(year + t_0)
                # then interpolate for given year
        else:
            h_norm = np.array(load.distribution.computeQuantile(1 - p_h))[0]
        input["h"] = h_norm

    if hasattr(load, "dist_change"):
        if isinstance(load.dist_change, float):  # for SAFE input
            # this is only for piping and stability. For overflow it should be extended with use of the HBN factor
            input["dh"] = load.dist_change * year
        else:
            dh = np.array(load.dist_change.computeQuantile(p_dh))[0]
            input["dh"] = dh * year
    else:
        input["dh"] = 0.0
    return input


###################################################################################################
## THESE ARE FASTER FORMULAS FOR CONVERTING BETA TO PROB AND VICE VERSA


def beta_to_pf(beta: float | list[float]) -> float | list[float]:
    # alternative: use scipy
    if isinstance(beta, list):
        return norm.cdf([-_element for _element in beta]).tolist()
    return norm.cdf(-beta)


def pf_to_beta(pf):
    # alternative: use scipy
    return -norm.ppf(pf)
