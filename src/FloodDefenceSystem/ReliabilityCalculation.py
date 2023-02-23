import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import pandas as pd
from scipy import interpolate

import src.FloodDefenceSystem.Mechanisms as Mechanisms
import src.ProbabilisticTools.ProbabilisticFunctions as ProbabilisticFunctions
from src.defaults.vrtool_config import VrtoolConfig
from src.ProbabilisticTools.HydraRing_scripts import (
    DesignTableOpenTurns,
    readDesignTable,
)
from src.ProbabilisticTools.ProbabilisticFunctions import (
    FragilityIntegration,
    IterativeFC_calculation,
    TableDist,
    TemporalProcess,
    addLoadCharVals,
    beta_to_pf,
    pf_to_beta,
    run_prob_calc,
)


class LoadInput:
    # class to store load data
    def __init__(self, section_fields):
        if "Load_2025" in section_fields:
            self.load_type = "HRING"
        elif "YearlyWLRise" in section_fields:
            self.load_type = "SAFE"

    def set_HRING_input(self, folder, section, gridpoints=1000):
        years = os.listdir(folder)
        self.distribution = {}
        for year in years:
            self.distribution[year] = DesignTableOpenTurns(
                folder.joinpath(
                    year, "{}.txt".format(getattr(section, "Load_{}".format(year)))
                ),
                gridpoints=gridpoints,
            )

    def set_fromDesignTable(self, filelocation, gridpoints=1000):
        # Load is given by exceedence probability-water level table from Hydra-Ring
        self.distribution = DesignTableOpenTurns(filelocation, gridpoints=gridpoints)

    def set_annual_change(self, type="determinist", parameters=[0]):
        # set an annual change of the water level
        if type == "determinist":
            self.dist_change = ot.Dirac(parameters)
        elif type == "SAFE":  # specific formulation for SAFE
            self.dist_change = parameters[0]
            self.HBN_factor = parameters[1]
        elif type == "gamma":
            self.dist_change = ot.Gamma()
            self.dist_change.setParameter(ot.GammaMuSigma()(parameters))

    def plot_load_cdf(self):
        data = np.array(self.distribution.getParameter())
        x = np.split(data, 2)
        plt.plot(x[0], 1 - x[1])
        plt.yscale("log")
        plt.title("Probability of non-exceedence")
        plt.xlabel("Water level [m NAP]")
        plt.ylabel(r"$P_{non exceedence} (-/year)$")


# A collection of MechanismReliability objects in time
class MechanismReliabilityCollection:
    def __init__(self, mechanism, computation_type, measure_year=0):
        # Initialize and make collection of MechanismReliability objects
        # mechanism, type, years are universal.
        # Measure_year is to indicate whether the reliability has to be recalculated or can be copied
        # (the latter is the case if a measure is taken later than the considered point in time)

        self.Reliability = {}

        for i in config.T:
            if measure_year > i:
                self.Reliability[str(i)] = MechanismReliability(
                    mechanism, computation_type, copy_or_calculate="copy"
                )
            else:
                self.Reliability[str(i)] = MechanismReliability(
                    mechanism, computation_type
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
                    self.Reliability[i].Input,
                    load,
                    mechanism=mechanism,
                    method=method,
                    year=float(i),
                    TrajectInfo=trajectinfo,
                )
                for i in self.Reliability.keys()
            ]
        else:
            [
                self.Reliability[i].calcReliability(
                    mechanism=mechanism,
                    method=method,
                    year=float(i),
                    TrajectInfo=trajectinfo,
                )
                for i in self.Reliability.keys()
            ]

        # NB: This could be extended with conditional failure probabilities

    def constructFragilityCurves(self, input, start=5, step=0.2):
        # Construct fragility curves for the entire collection
        for i in self.Reliability.keys():
            self.Reliability[i].constructFragilityCurve(
                self.Reliability[i].mechanism, input, year=i, start=start, step=step
            )

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
            t.append(float(i) + config.t_0)
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


class MechanismReliability:
    # This class contains evaluations of the reliability for a mechanism in a given year.
    def __init__(self, mechanism, type, copy_or_calculate="calculate"):
        # Initialize: set mechanism and type. These are the most important basic parameters
        self.mechanism = mechanism
        self.type = type
        self.copy_or_calculate = copy_or_calculate
        self.Input = MechanismInput(self.mechanism)
        if mechanism == "Piping":
            self.gamma_schem_heave = 1  # 1.05
            self.gamma_schem_upl = 1  # 1.05
            self.gamma_schem_pip = 1  # 1.05
        else:
            pass

    def __clearvalues__(self):
        # clear all values
        keys = self.__dict__.keys()
        for i in keys:
            # if i is not 'mechanism':
            if i != "mechanism":
                setattr(self, i, None)

    def constructFragilityCurve(
        self,
        mechanism,
        input,
        start=5,
        step=0.2,
        method="MCS",
        splitPiping="no",
        year=0,
        lolim=10e-4,
        hilim=0.995,
    ):
        # Script to construct fragility curves from a mechanism model.
        if mechanism == "Piping":
            # initialize the required lists
            marginals = []
            names = []
            if splitPiping == "yes":
                Pheave = []
                Puplift = []
                Ppiping = []
                result_heave = []
                result_uplift = []
                result_piping = []
                wl_heave = []
                wl_uplift = []
                wl_piping = []
            else:
                Ptotal = []
                result_total = []
                wl_total = []

            # make list of all random variables:
            for i in input.input.keys():
                marginals.append(input.input[i])
                names.append(i)

            marginals.append(ot.Dirac(1.0))
            names.append("h")

            if splitPiping == "yes":
                result_heave, Pheave, wl_heave = IterativeFC_calculation(
                    marginals,
                    start,
                    names,
                    Mechanisms.zHeave,
                    method,
                    step,
                    lolim,
                    hilim,
                )
                result_piping, Ppiping, wl_piping = IterativeFC_calculation(
                    marginals,
                    start,
                    names,
                    Mechanisms.zPiping,
                    method,
                    step,
                    lolim,
                    hilim,
                )
                result_uplift, Puplift, wl_uplift = IterativeFC_calculation(
                    marginals,
                    start,
                    names,
                    Mechanisms.zUplift,
                    method,
                    step,
                    lolim,
                    hilim,
                )
                self.h_cUplift = ot.Distribution(
                    TableDist(np.array(wl_uplift), np.array(Puplift), extrap=True)
                )
                self.resultsUplift = result_uplift
                self.wlUplift = wl_uplift
                self.h_cHeave = ot.Distribution(
                    TableDist(np.array(wl_heave), np.array(Pheave), extrap=True)
                )
                self.resultsHeave = result_heave
                self.wlHeave = wl_heave
                self.h_cPiping = ot.Distribution(
                    TableDist(np.array(wl_piping), np.array(Ppiping), extrap=True)
                )
                self.resultsPiping = result_piping
                self.wlPiping = wl_piping

            if splitPiping == "no":
                result_total, Ptotal, wl_total = IterativeFC_calculation(
                    marginals,
                    start,
                    names,
                    Mechanisms.zPipingTotal,
                    method,
                    step,
                    lolim,
                    hilim,
                )
                self.h_c = ot.Distribution(
                    TableDist(np.array(wl_total), np.array(Ptotal), extrap=True)
                )
                self.results = result_total
                self.wl = wl_total

        elif mechanism == "Overflow":
            # make list of all random variables:
            marginals = []
            names = []

            for i in input.input.keys():
                # Adapt temporal process variables
                if i in input.temporals:
                    # Possibly this can be done using an OpenTurns random walk object
                    original = copy.deepcopy(input.input[i])
                    adapt_dist = TemporalProcess(original, year)
                    marginals.append(adapt_dist)
                    names.append(i)
                else:
                    marginals.append(input.input[i])
                    names.append(i)

            result = []
            marginals.append(ot.Dirac(1.0))
            names.append("h")  # add the load
            result, P, wl = IterativeFC_calculation(
                marginals,
                start,
                names,
                Mechanisms.zOverflow,
                method,
                step,
                lolim,
                hilim,
            )
            self.h_c = ot.Distribution(
                TableDist(np.array(wl), np.array(P), extrap=True)
            )
            self.results = result
            self.wl = wl
        elif mechanism == "StabilityInner":
            pass

        self.type = "FragilityCurve"

    def calcReliability(
        self,
        strength=False,
        load=False,
        mechanism=None,
        method="FORM",
        year=0,
        TrajectInfo=None,
    ):
        # This routine calculates cross-sectional reliability indices based on different types of calculations.
        if self.type == "DirectInput":
            t_grid = list(self.Input.input["beta"].keys())
            beta_grid = list(self.Input.input["beta"].values())
            betat = interpolate.interp1d(t_grid, beta_grid, fill_value="extrapolate")
            beta = np.float32(betat(year))
            self.beta = beta
            self.Pf = beta_to_pf(self.beta)

        if self.type == "HRING":
            if mechanism == "Overflow":
                self.beta, self.Pf = Mechanisms.OverflowHRING(self.Input.input, year)
            else:
                raise Exception(
                    "Unknown computation type HRING for {}".format(mechanism)
                )
        if self.type == "Simple":
            if mechanism == "StabilityInner":
                if "SF_2025" in strength.input:
                    # Simple interpolation of two safety factors and translation to a value of beta at 'year'.
                    # In this model we do not explicitly consider climate change, as it is already in de SF estimates by Sweco
                    SFt = interpolate.interp1d(
                        [0, 50],
                        np.array(
                            [strength.input["SF_2025"], strength.input["SF_2075"]]
                        ).flatten(),
                        fill_value="extrapolate",
                    )
                    SF = SFt(year)
                    beta = np.min([beta_SF_StabilityInner(SF, type="SF"), 8.0])
                elif "beta_2025" in strength.input:

                    betat = interpolate.interp1d(
                        [0, 50],
                        np.array(
                            [strength.input["beta_2025"], strength.input["beta_2075"]]
                        ).flatten(),
                        fill_value="extrapolate",
                    )

                    beta = betat(year)
                    beta = np.min([beta, 8])
                elif (
                    "BETA" in strength.input
                ):  # situation where beta is constant in time
                    beta = np.min([strength.input["BETA"].item(), 8.0])
                else:
                    raise Exception(
                        "Warning: No input values SF or Beta StabilityInner"
                    )
                # Check if there is an elimination measure present (diaphragm wall)
                if "Elimination" in strength.input.keys():
                    if strength.input["Elimination"] == "yes":
                        # Fault tree: Pf = P(f|elimination fails)*P(elimination fails) + P(f|elimination works)* P(elimination works)
                        # addition: should not be more unsafe
                        self.Pf = np.min(
                            [
                                beta_to_pf(beta) * strength.input["Pf_elim"]
                                + strength.input["Pf_with_elim"]
                                * (1 - strength.input["Pf_elim"]),
                                beta_to_pf(beta),
                            ]
                        )
                        self.beta = pf_to_beta(self.Pf)
                    else:
                        raise ValueError(
                            "Warning: Elimination defined but not turned on"
                        )
                else:
                    self.beta = beta
                    self.Pf = beta_to_pf(self.beta)

            elif mechanism == "Overflow":  # specific for SAFE
                # climate change included, including a factor for HBN
                if hasattr(load, "dist_change"):
                    h_t = (
                        strength.input["h_crest"]
                        - (
                            strength.input["dhc(t)"]
                            + (load.dist_change * load.HBN_factor)
                        )
                        * year
                    )
                else:
                    h_t = strength.input["h_crest"] - (strength.input["dhc(t)"] * year)

                self.beta, self.Pf = Mechanisms.OverflowSimple(
                    h_t,
                    strength.input["q_crest"],
                    strength.input["h_c"],
                    strength.input["q_c"],
                    strength.input["beta"],
                    mode="assessment",
                )
            elif mechanism == "Piping":
                pass

            self.alpha_sq = np.nan
            self.result = np.nan
        elif self.type == "FragilityCurve":
            # Generic function for evaluating a fragility curve with water level and change in water level (optional)
            if hasattr(load, "dist_change"):
                original = copy.deepcopy(load.dist_change)
                dist_change = TemporalProcess(original, year)
                # marginals = [self.Input.input['FC'], load, dist_change]
                P, beta = FragilityIntegration(
                    self.Input.input["FC"], load, WaterLevelChange=dist_change
                )

                # result missing
                # dist = ot.ComposedDistribution(marginals)
                # dist.setDescription(['h_c', 'h', 'dh'])
                # TODO replace with FragilityIntegration. FragilityIntegration in ProbabilisticFunctions.py
                self.alpha_sq = np.nan
                self.result = np.nan
            else:
                marginals = [self.h_c, load.distribution]
                dist = ot.ComposedDistribution(marginals)
                dist.setDescription(["h_c", "h"])
                result, P, beta, alfas_sq = run_prob_calc(
                    ot.SymbolicFunction(["h_c", "h"], ["h_c-h"]), dist, method
                )
                self.result = result
                self.alpha_sq = alfas_sq
            self.Pf = P
            self.beta = beta

        elif self.type == "Prob":
            # Probabilistic evaluation of a mechanism.
            if mechanism == "Piping":
                zFunc = Mechanisms.zPipingTotal
            elif mechanism == "Overflow":
                zFunc = Mechanisms.zOverflow
            elif mechanism == "simpleLSF":
                zFunc = Mechanisms.simpleLSF
            else:
                raise ValueError("Unknown Z-function")

            if hasattr(self.Input, "char_vals"):
                start_vals = []
                for i in descr:
                    if i != "h" and i != "dh":
                        start_vals.append(
                            strength.char_vals[i]
                        ) if i not in strength.temporals else start_vals.append(
                            strength.char_vals[i] * year
                        )
                start_vals = addLoadCharVals(start_vals, load)
            else:
                start_vals = self.Input.input.getMean()

            result, P, beta, alpha_sq = run_prob_calc(
                ot.PythonFunction(self.Input.input.getDimension(), 1, zFunc),
                self.Input.input,
                method,
                startpoint=start_vals,
            )
            self.result = result
            self.Pf = P
            self.beta = beta
            self.alpha_sq = alpha_sq
        elif self.type == "SemiProb":
            # semi probabilistic assessment, only available for piping
            if mechanism == "Piping":
                if TrajectInfo == None:  # Defaults, typical values for 16-3 and 16-4
                    TrajectInfo = {}
                    TrajectInfo["Pmax"] = 1.0 / 10000
                    TrajectInfo["omegaPiping"] = 0.24
                    TrajectInfo["bPiping"] = 300
                    TrajectInfo["aPiping"] = 0.9
                    TrajectInfo["TrajectLength"] = 20000
                # First calculate the SF without gamma for the three submechanisms
                # Piping:
                strength_new = copy.deepcopy(strength)
                self.scenario_result = {}
                self.scenario_result["Scenario"] = strength_new.input["Scenario"]
                self.scenario_result["P_scenario"] = strength_new.input["P_scenario"]
                self.scenario_result["beta_cs_p"] = {}
                self.scenario_result["beta_cs_h"] = {}
                self.scenario_result["beta_cs_u"] = {}
                self.scenario_result["Pf"] = {}
                self.scenario_result["Beta"] = {}

                for i in strength.temporals:
                    strength_new.input[i] = strength.input[i] * year

                # TODO:below, remove self. in for example self.gamma_pip. This is just an scenario output value. do not store.
                # calculate beta per scenario and determine overall
                for j in range(0, len(strength_new.input["Scenario"])):
                    strength_new.input_ind = {}
                    for i in strength_new.input:  # select values of scenario j
                        try:
                            strength_new.input_ind[i] = strength_new.input[i][j]
                        except:
                            pass  # TODO: make more clean, na measures doorloopt hij deze loop nogmaals, niet voor alle variabelen in strength_new.input is een array beschikbaar.

                    # inputs = addLoadCharVals(strength_new.input, load=None, p_h=TrajectInfo['Pmax'], p_dh=0.5, year=year)
                    # inputs['h'] = load.NormWaterLevel
                    # TODO aanpassen met nieuwe belastingmodel
                    inputs = addLoadCharVals(
                        strength_new.input_ind,
                        load=load,
                        p_h=TrajectInfo["Pmax"],
                        p_dh=0.5,
                        year=year,
                    )

                    Z, self.p_dh, self.p_dh_c = Mechanisms.zPiping(
                        inputs, mode="SemiProb"
                    )
                    self.gamma_pip = TrajectInfo["gammaPiping"]
                    # ProbabilisticFunctions.calc_gamma('Piping', TrajectInfo=TrajectInfo) #
                    # Calculate needed safety factor

                    if self.p_dh != 0:
                        self.SF_p = (
                            self.p_dh_c / (self.gamma_pip * self.gamma_schem_pip)
                        ) / self.p_dh
                    else:
                        self.SF_p = np.inf
                    self.assess_p = "voldoende" if self.SF_p > 1 else "onvoldoende"
                    self.scenario_result["beta_cs_p"][
                        j
                    ] = ProbabilisticFunctions.calc_beta_implicated(
                        "Piping", self.SF_p * self.gamma_pip, TrajectInfo=TrajectInfo
                    )  #
                    # Calculate the implicated beta_cs

                    # Heave:
                    Z, self.h_i, self.h_i_c = Mechanisms.zHeave(inputs, mode="SemiProb")
                    self.gamma_h = TrajectInfo[
                        "gammaHeave"
                    ]  # ProbabilisticFunctions.calc_gamma('Heave',TrajectInfo=TrajectInfo)  #
                    # Calculate
                    # needed safety factor
                    # TODO: check formula Sander Kapinga
                    self.SF_h = (
                        self.h_i_c / (self.gamma_schem_heave * self.gamma_h)
                    ) / self.h_i
                    self.assess_h = (
                        "voldoende"
                        if (self.h_i_c / (self.gamma_schem_heave * self.gamma_h))
                        / self.h_i
                        > 1
                        else "onvoldoende"
                    )
                    self.scenario_result["beta_cs_h"][
                        j
                    ] = ProbabilisticFunctions.calc_beta_implicated(
                        "Heave",
                        (self.h_i_c / self.gamma_schem_heave) / self.h_i,
                        TrajectInfo=TrajectInfo,
                    )  # Calculate the implicated beta_cs

                    # Uplift
                    Z, self.u_dh, self.u_dh_c = Mechanisms.zUplift(
                        inputs, mode="SemiProb"
                    )
                    self.gamma_u = TrajectInfo[
                        "gammaUplift"
                    ]  # ProbabilisticFunctions.calc_gamma('Uplift',TrajectInfo=TrajectInfo)
                    # Calculate
                    # needed safety factor
                    # TODO: check formula Sander Kapinga
                    self.SF_u = (
                        self.u_dh_c / (self.gamma_schem_upl * self.gamma_u)
                    ) / self.u_dh

                    self.assess_u = (
                        "voldoende"
                        if (self.u_dh_c / (self.gamma_schem_upl * self.gamma_u))
                        / self.u_dh
                        > 1
                        else "onvoldoende"
                    )
                    self.scenario_result["beta_cs_u"][
                        j
                    ] = ProbabilisticFunctions.calc_beta_implicated(
                        "Uplift",
                        (self.u_dh_c / self.gamma_schem_upl) / self.u_dh,
                        TrajectInfo=TrajectInfo,
                    )  # Calculate the implicated beta_cs

                    # Check if there is an elimination measure present (VZG or diaphragm wall)
                    if "Elimination" in strength.input.keys():
                        if strength.input["Elimination"] == "yes":
                            # Fault tree: Pf = P(f|elimination fails)*P(elimination fails) + P(f|elimination works)* P(elimination works)
                            scenario_beta = np.max(
                                [
                                    self.scenario_result["beta_cs_h"][j],
                                    self.scenario_result["beta_cs_u"][j],
                                    self.scenario_result["beta_cs_p"][j],
                                ]
                            )
                            self.scenario_result["Pf"][j] = np.min(
                                [
                                    np.min(
                                        [
                                            beta_to_pf(scenario_beta)
                                            * strength.input["Pf_elim"]
                                            + strength.input["Pf_with_elim"]
                                            * (1 - strength.input["Pf_elim"]),
                                            beta_to_pf(scenario_beta),
                                        ]
                                    ),
                                    beta_to_pf(8.0),
                                ]
                            )
                            self.scenario_result["Beta"][j] = np.min(
                                [pf_to_beta(self.scenario_result["Pf"][j]), 8.0]
                            )

                        else:
                            raise ValueError(
                                "Warning: Elimination defined but not turned on"
                            )
                    else:
                        self.scenario_result["Beta"][j] = np.min(
                            [
                                np.max(
                                    [
                                        self.scenario_result["beta_cs_h"][j],
                                        self.scenario_result["beta_cs_u"][j],
                                        self.scenario_result["beta_cs_p"][j],
                                    ]
                                ),
                                8,
                            ]
                        )
                        self.scenario_result["Pf"][j] = beta_to_pf(
                            self.scenario_result["Beta"][j]
                        )

                # multiply every scenario by probability
                self.Pf = np.max(
                    [
                        sum(
                            self.scenario_result["Pf"][k]
                            * self.scenario_result["P_scenario"][k]
                            for k in self.scenario_result["Pf"]
                        ),
                        beta_to_pf(8.0),
                    ]
                )
                self.Beta = np.min([pf_to_beta(self.Pf), 8])

                self.WLchar = copy.deepcopy(
                    inputs["h"]
                )  # add water level as used in the assessment
                self.alpha_sq = np.nan
                self.result = np.nan
                # if year == 50:
                #      print(year, self.beta_cs_u ,self.beta_cs_h,self.beta_cs_p, self.WLchar)

            else:
                pass


class MechanismInput:
    # Class for input of a mechanism
    def __init__(self, mechanism):
        self.mechanism = mechanism

    def fill_distributions(self, distributions, t, processIDs, parameters):

        dists = copy.deepcopy(distributions)
        self.temporals = []
        for j in processIDs:
            if t > 0:
                dists[j] = TemporalProcess(dists[j], t)
            else:
                dists[j] = ot.Dirac(0.0)
            self.temporals.append(parameters[j])
        self.input = ot.ComposedDistribution(dists)
        self.input.setDescription(parameters)

    # This routine reads  input from an input sheet
    def fill_mechanism(
        self,
        input_path,
        reference,
        calctype,
        kind="csv",
        sheet=None,
        mechanism=None,
        **kwargs,
    ):
        if kind == "csv":
            if mechanism != "Overflow":
                try:
                    data = pd.read_csv(
                        input_path.joinpath(Path(str(reference)).name),
                        delimiter=",",
                        header=None,
                    )
                except:
                    data = pd.read_csv(
                        input_path.joinpath(Path(str(reference)).name + ".csv"),
                        delimiter=",",
                        header=None,
                    )

                # TODO: fix datatypes in input such that we do not need to drop columns
                data = data.rename(columns={list(data)[0]: "Name"})
                data = data.set_index("Name")
                try:
                    data = data.drop(["InScope", "Opmerking"]).astype(np.float32)
                except:
                    pass

            else:  #'Overflow':
                if calctype == "Simple":
                    data = pd.read_csv(
                        input_path.joinpath(Path(reference).name), delimiter=","
                    )
                    data = data.transpose()
                elif calctype == "HRING":
                    # detect years
                    years = os.listdir(input_path)
                    for count, year in enumerate(years):
                        year_data = readDesignTable(
                            input_path.joinpath(year, reference + ".txt")
                        )[["Value", "Beta"]]
                        if count == 0:
                            data = year_data.set_index("Value").rename(
                                columns={"Beta": year}
                            )

                        else:
                            if all(data.index.values == year_data.Value.values):
                                data = pd.concat(
                                    (
                                        data,
                                        year_data.set_index("Value").rename(
                                            columns={"Beta": year}
                                        ),
                                    ),
                                    axis="columns",
                                )
                            # compare value columns:
                        # if count>0 and Value is identical: concatenate.
                        # else: interpolate and then concatenate.
                else:
                    raise Exception("Unknown input type for overflow")

        elif kind == "xlsx":
            data = pd.read_excel(input, sheet_name=sheet)
            data = data.set_index("Name")

        self.input = {}
        self.temporals = []
        self.char_vals = {}
        for i in range(len(data)):
            # if (data.iloc[i].Name == 'FragilityCurve') and ~np.isnan(data.iloc[i].Value):
            if data.index[i] == "FragilityCurve":
                pass
                # Turned off: old code that doesnt work anymore
            elif calctype == "HRING":
                self.input["hc_beta"] = data
                self.input["h_crest"] = kwargs["crest_height"]
                self.input["d_crest"] = kwargs["dcrest"]
            else:
                x = data.iloc[i][:].values
                if isinstance(x, np.ndarray):
                    if len(x) > 1:
                        self.input[data.index[i]] = x.astype(np.float32)[~np.isnan(x)]
                    elif len(x) == 1:
                        try:
                            if not np.isnan(np.float32(x[0])):
                                self.input[data.index[i]] = np.array([np.float32(x[0])])
                        except:
                            self.input[data.index[i]] = x[0]
                    else:
                        pass
                else:
                    pass

                if data.index[i][-3:] == "(t)":
                    self.temporals.append(data.index[i])

                # for k-value: ensure that value is in m/s not m/d:
                if data.index[i] == "k":
                    try:
                        if any(
                            self.input[data.index[i]] > 1.0
                        ):  # if k>1 it is likely in m/d
                            self.input[data.index[i]] = self.input[data.index[i]] / (
                                24 * 3600
                            )
                            print(
                                "k-value modified as it was likely m/d and should be m/s"
                            )
                    except:
                        if self.input[data.index[i]] > 1.0:
                            self.input[data.index[i]] = self.input[data.index[i]] / (
                                24 * 3600
                            )
                            print(
                                "k-value modified as it was likely m/d and should be m/s"
                            )


def beta_SF_StabilityInner(SF_or_beta, type=False, modelfactor=1.06):
    """Careful: ensure that upon using this function you clearly define the input parameter!"""
    if type == "SF":
        beta = ((SF_or_beta.item() / modelfactor) - 0.41) / 0.15
        beta = np.min([beta, 8.0])
        return beta
    elif type == "beta":
        SF = (0.41 + 0.15 * SF_or_beta) * modelfactor
        return SF
