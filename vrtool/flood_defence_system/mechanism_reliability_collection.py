import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import pandas as pd

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


# A collection of MechanismReliability objects in time
class MechanismReliabilityCollection:
    def __init__(
        self, mechanism, computation_type, config: VrtoolConfig, measure_year=0
    ):
        # Initialize and make collection of MechanismReliability objects
        # mechanism, type, years are universal.
        # Measure_year is to indicate whether the reliability has to be recalculated or can be copied
        # (the latter is the case if a measure is taken later than the considered point in time)
        self.T = config.T
        self.t_0 = config.t_0
        self.Reliability = {}

        for i in self.T:
            if measure_year > i:
                self.Reliability[str(i)] = MechanismReliability(
                    mechanism, computation_type, self.t_0, copy_or_calculate="copy"
                )
            else:
                self.Reliability[str(i)] = MechanismReliability(
                    mechanism, computation_type, self.t_0
                )

    def generateInputfromDistributions(
        self, distributions, parameters=["R", "dR", "S"], processes=["dR"]
    ):
        processIDs = []
        for process in processes:
            processIDs.append(parameters.index(process))

        for i in self.Reliability:
            self.Reliability[i].Input.fill_distributions(
                distributions, np.int32(i), processIDs, parameters
            )

            pass

    def generateLCRProfile(
        self,
        load=False,
        mechanism="Overflow",
        method="FORM",
        trajectinfo=None,
        interpolate="False",
        conditionality="no",
    ):
        # this function generates life-cycle reliability based on the years that have been calculated (so reliability in time)
        if load:
            [
                self.Reliability[i].calcReliability(
                    mechanism=mechanism,
                    year=float(i),
                    traject_info=trajectinfo,
                    strength = self.Reliability[i].Input,
                    load = load,
                )
                for i in self.Reliability.keys()
            ]
        else:
            [
                self.Reliability[i].calcReliability(
                    mechanism=mechanism,
                    year=float(i),
                    traject_info=trajectinfo,
                )
                for i in self.Reliability.keys()
            ]

        # NB: This could be extended with conditional failure probabilities

    def calcLifetimeProb(self, conditionality="no", period=None):
        # This script calculates the total probability over a certain period. It assumes independence of years.
        # This can be improved in the future to account for correlation.
        years = list(self.Reliability.keys())
        # set grid to range of calculations or defined period:
        tgrid = (
            np.arange(np.int8(years[0]), np.int8(years[-1:]) + 1, 1)
            if period == None
            else np.arange(np.int8(years[0]), period, 1)
        )
        t0 = []
        beta0 = []

        for i in years:
            t0.append(np.int8(i))
            beta0.append(self.Reliability[i].beta)

        # calculate beta's per year, transform to pf, accumulate and then calculate beta for the period
        beta = np.interp(tgrid, t0, beta0)
        pfs = beta_to_pf(beta)
        pftot = 1 - np.cumprod(1 - pfs)
        self.beta_life = (np.max(tgrid), np.float(pf_to_beta(pftot[-1:])))

    def getProbinYear(self, year):
        # Interpolate a beta in a defined year from a collection of beta values
        t0 = []
        beta0 = []
        years = list(self.Reliability.keys())

        for i in years:
            t0.append(np.int8(i))
            beta0.append(self.Reliability[i].beta)

        beta = np.interp(year, t0, beta0)
        return beta

    def drawLCR(self, yscale=None, type="beta", mechanism=None):
        # Draw the life cycle reliability. Default is beta but can be set to Pf
        t = []
        y = []

        for i in self.Reliability.keys():
            t.append(float(i) + self.t_0)
            if self.Reliability[i].type == "Probabilistic":
                if self.Reliability[i].result.getClassName() == "SimulationResult":
                    y.append(
                        self.Reliability[i].result.getProbabilityEstimate()
                    ) if type == "pf" else y.append(
                        -ot.Normal().computeScalarQuantile(
                            self.Reliability[i].result.getProbabilityEstimate()
                        )
                    )
                else:
                    y.append(
                        self.Reliability[i].result.getEventProbability()
                    ) if type == "pf" else y.append(
                        self.Reliability[i].result.getHasoferReliabilityIndex()
                    )
            else:
                y.append(self.Reliability[i].Pf) if type == "pf" else y.append(
                    self.Reliability[i].beta
                )

        plt.plot(t, y, label=mechanism)
        if yscale == "log":
            plt.yscale(yscale)

        plt.xlabel("Time")
        plt.ylabel(r"$\beta$") if type != "pf" else plt.ylabel(r"$P_f$")
        plt.title("Life-cycle reliability")

    def drawFC(self, yscale=None):
        # Drawa a fragility curve
        for j in self.Reliability.keys():
            wl = self.Reliability[j].wl
            pf = [
                self.Reliability[j].results[i].getProbabilityEstimate()
                for i in range(0, len(self.Reliability[j].results))
            ]
            plt.plot(wl, pf, label=j)

        plt.legend()
        plt.ylabel("Pf|h[-/year]")
        plt.xlabel("h[m +NAP]")
        if yscale == "log":
            plt.yscale("log")

    def drawAlphaBar(self, step=5):
        import matplotlib.ticker as ticker

        alphas = np.array([])
        firstKey = list(self.Reliability.keys())[0]
        alphaDim = len(self.Reliability[firstKey].alpha_sq)
        alphas = np.concatenate(
            [self.Reliability[i].alpha_sq for i in self.Reliability.keys()]
        )
        alphas = np.reshape(alphas, (np.int(np.size(alphas) / alphaDim), alphaDim))
        variableNames = list(self.Reliability[firstKey].Input.input.getDescription())
        alphas = pd.DataFrame(alphas, columns=variableNames)
        ax = alphas.plot.bar(stacked=True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
        plt.ylim([0, 1])
        plt.title(r"Influence coefficients $\alpha$ in time")
        plt.show()
