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


def run_prob_calc(model, dist, method="FORM", startpoint=False):
    vect = ot.RandomVector(dist)
    if method == "MCS":
        model = ot.MemoizeFunction(model)
    G = ot.CompositeRandomVector(model, vect)
    event = ot.Event(G, ot.Less(), 0)

    if method == "FORM":
        solver = ot.AbdoRackwitz()
        solver.setMaximumAbsoluteError(1e-2)
        solver.setMaximumRelativeError(1e-2)
        solver.setMaximumIterationNumber(200)
        if startpoint:
            algo = ot.FORM(solver, event, startpoint)
        else:
            algo = ot.FORM(solver, event, dist.getMean())

        # else:
        #     startpoint = 0
        algo.run()
        result = algo.getResult()
        Pf = result.getEventProbability()
        beta = result.getHasoferReliabilityIndex()
        alfas_sq = (np.array(result.getStandardSpaceDesignPoint()) / beta) ** 2
    elif method == "DIRS":
        result, algo = run_DIRS(event, approach=ot.MediumSafe(), samples=1000)
        Pf = result.getProbabilityEstimate()
    elif method == "MCS":
        ot.RandomGenerator.SetSeed(5000)
        logging.warn("Random Generator state is currently fixed!")
        experiment = ot.MonteCarloExperiment()
        algo = ot.ProbabilitySimulationAlgorithm(event, experiment)
        algo.setMaximumCoefficientOfVariation(0.05)
        algo.setMaximumStandardDeviation(1e-3)
        algo.setMaximumOuterSampling(int(1000))
        algo.setBlockSize(100)
        algo.run()
        result = algo.getResult()
        Pf = result.getProbabilityEstimate()
        beta = pf_to_beta(Pf)
        if beta != float("Inf"):
            alfas_sq = np.array(result.getImportanceFactors())
        else:
            alfas_sq = np.empty(
                (
                    1,
                    vect.getDimension(),
                )
            )
            alfas_sq[:] = np.nan
    return result, Pf, beta, alfas_sq


def run_DIRS(
    event, approach=ot.MediumSafe(), sampling=ot.OrthogonalDirection(), samples=250
):
    # start = time.time()
    algo = ot.DirectionalSampling(event, approach, sampling)
    algo.setMaximumOuterSampling(samples)
    algo.setBlockSize(4)
    algo.setMaximumCoefficientOfVariation(0.025)
    algo.run()
    result = algo.getResult()
    probability = result.getProbabilityEstimate()
    # end = time.time()
    logging.info(event.getName())
    logging.info("%15s" % "Pf = ", "{0:.2E}".format(probability))
    logging.info(
        "%15s" % "CoV = ", "{0:.2f}".format(result.getCoefficientOfVariation())
    )
    logging.info("%15s" % "N = ", "{0:.0f}".format(result.getOuterSampling()))
    return result, algo


def iterative_fc_calculation(
    marginals, WL, names, zFunc, method, step=0.5, lolim=10e-4, hilim=0.999
):
    marginals[len(marginals) - 1] = ot.Dirac(float(WL))
    dist = ot.ComposedDistribution(marginals)
    dist.setDescription(names)
    result, P, beta, alpha = run_prob_calc(
        ot.PythonFunction(len(marginals), 1, zFunc), dist, method
    )
    wl_list = []
    result_list = []
    P_list = []
    while P > hilim or P < lolim:
        logging.info("changed start value")
        WL = WL - 1 if P > hilim else WL + 1
        marginals[len(marginals) - 1] = ot.Dirac(float(WL))
        dist = ot.ComposedDistribution(marginals)
        dist.setDescription(names)
        result, P, beta, alpha = run_prob_calc(
            ot.PythonFunction(len(marginals), 1, zFunc), dist, method
        )

    result_list.append(result)
    wl_list.append(WL)
    P_list.append(P)
    count = 0
    while P > lolim:
        WL -= step
        count += 1
        marginals[len(marginals) - 1] = ot.Dirac(float(WL))
        dist = ot.ComposedDistribution(marginals)
        dist.setDescription(names)
        result, P, beta, alpha = run_prob_calc(
            ot.PythonFunction(len(marginals), 1, zFunc), dist, method
        )
        result_list.append(result)
        wl_list.append(WL)
        P_list.append(P)
        logging.info(str(count) + " calculations made for fragility curve")
    WL = max(wl_list)
    while P < hilim:
        WL += step
        count += 1
        marginals[len(marginals) - 1] = ot.Dirac(float(WL))
        dist = ot.ComposedDistribution(marginals)
        dist.setDescription(names)
        result, P, beta, alpha = run_prob_calc(
            ot.PythonFunction(len(marginals), 1, zFunc), dist, method
        )
        result_list.append(result)
        wl_list.append(WL)
        P_list.append(P)
        logging.info(str(count) + " calculations made for fragility curve")

    indices = list(np.argsort(wl_list))
    wl_list = [wl_list[i] for i in indices]
    result_list = [result_list[i] for i in indices]
    P_list = [P_list[i] for i in indices]
    indexes = np.where(np.diff(P_list) == 0)
    rm_items = 0
    if len(indexes[0]) > 0:
        for i in indexes[0]:
            wl_list.pop(i - rm_items)
            result_list.pop(i - rm_items)
            P_list.pop(i - rm_items)
            rm_items += 1
    # remove the non increasing values
    return result_list, P_list, wl_list


def temporal_process(temporal_input, t, makePlot="off"):
    # TODO check of input == float.
    if isinstance(temporal_input, float):
        temporal_input = ot.Dirac(temporal_input * t)  # make distribution
    # This function derives the distribution parameters for the temporal process governed by the annual distribution 'input' for year 't'
    elif temporal_input.getClassName() == "Gamma":
        params = temporal_input.getParameter()
        mu = params[0] / params[1]
        var = params[0] / (params[1] ** 2)
        temporal_input.setParameter(
            ot.GammaMuSigma()([mu * float(t), np.sqrt(var) * float(t), 0])
        )
        if makePlot == "on":
            gr = temporal_input.drawPDF()
            from openturns.viewer import View

            view = View(gr)
            view.show()
    elif temporal_input.getClassName() == "Dirac":
        temporal_input.setParameter(temporal_input.getParameter() * t)
    else:
        raise Exception("Distribution type for temporal process not recognized.")
    return temporal_input


def get_design_water_level(load, p):
    return np.array(load.distribution.computeQuantile(1 - p))[0]


def add_load_char_vals(
    input, t_0: int, load=None, p_h=1.0 / 1000, p_dh=0.5, year=0
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
                for j in years:
                    wls.append(load.distribution[str(j)].computeQuantile(1 - p_h)[0])
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
            p = 0.5
            dh = np.array(load.dist_change.computeQuantile(p_dh))[0]
            input["dh"] = dh * year
    else:
        input["dh"] = 0.0
    return input


###################################################################################################
## THESE ARE FASTER FORMULAS FOR CONVERTING BETA TO PROB AND VICE VERSA


def beta_to_pf(beta):
    # alternative: use scipy
    return norm.cdf(-beta)


def pf_to_beta(pf):
    # alternative: use scipy
    return -norm.ppf(pf)
