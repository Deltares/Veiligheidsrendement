import copy

import numpy as np
import openturns as ot
from scipy import interpolate

import vrtool.flood_defence_system.mechanisms as fds_mechanisms
from vrtool.failure_mechanisms.overflow.overflow import Overflow
from vrtool.failure_mechanisms.stability_inner.stability_inner import StabilityInner
from vrtool.flood_defence_system.mechanism_input import MechanismInput
from vrtool.probabilistic_tools.probabilistic_functions import (
    TableDist,
    add_load_char_vals,
    beta_to_pf,
    calc_beta_implicated,
    calculate_fragility_integration,
    iterative_fc_calculation,
    pf_to_beta,
    run_prob_calc,
    temporal_process,
)

from vrtool.failure_mechanisms.general.direct_failure_mechanism import (
    calculate_reliability,
)


class MechanismReliability:
    # This class contains evaluations of the reliability for a mechanism in a given year.
    def __init__(self, mechanism, type, t_0: int, copy_or_calculate="calculate"):
        # Initialize: set mechanism and type. These are the most important basic parameters
        self.mechanism = mechanism
        self.type = type
        self.t_0 = t_0
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
                result_heave, Pheave, wl_heave = iterative_fc_calculation(
                    marginals,
                    start,
                    names,
                    fds_mechanisms.calculate_z_heave,
                    method,
                    step,
                    lolim,
                    hilim,
                )
                result_piping, Ppiping, wl_piping = iterative_fc_calculation(
                    marginals,
                    start,
                    names,
                    fds_mechanisms.calculate_z_piping,
                    method,
                    step,
                    lolim,
                    hilim,
                )
                result_uplift, Puplift, wl_uplift = iterative_fc_calculation(
                    marginals,
                    start,
                    names,
                    fds_mechanisms.calculate_z_uplift,
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
                result_total, Ptotal, wl_total = iterative_fc_calculation(
                    marginals,
                    start,
                    names,
                    fds_mechanisms.calculate_z_piping_total,
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
                    adapt_dist = temporal_process(original, year)
                    marginals.append(adapt_dist)
                    names.append(i)
                else:
                    marginals.append(input.input[i])
                    names.append(i)

            result = []
            marginals.append(ot.Dirac(1.0))
            names.append("h")  # add the load
            result, P, wl = iterative_fc_calculation(
                marginals,
                start,
                names,
                fds_mechanisms.calculate_z_overflow,
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
            self.beta, self.Pf = calculate_reliability(t_grid, beta_grid, year)

        if self.type == "HRING":
            if mechanism == "Overflow":
                self.beta, self.Pf = Overflow.overflow_hring(
                    self.Input.input, year, self.t_0
                )
            else:
                raise Exception(
                    "Unknown computation type HRING for {}".format(mechanism)
                )
        if self.type == "Simple":
            if mechanism == "StabilityInner":
                (
                    self.beta,
                    self.Pf,
                ) = StabilityInner.calculate_simple(strength, year)

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

                self.beta, self.Pf = Overflow.overflow_simple(
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
                dist_change = temporal_process(original, year)
                # marginals = [self.Input.input['FC'], load, dist_change]
                P, beta = calculate_fragility_integration(
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
                zFunc = fds_mechanisms.calculate_z_piping_total
            elif mechanism == "Overflow":
                zFunc = fds_mechanisms.calculate_z_overflow
            elif mechanism == "simpleLSF":
                zFunc = fds_mechanisms.calculate_simple_lsf
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
                start_vals = add_load_char_vals(start_vals, self.t_0, load)
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
                    inputs = add_load_char_vals(
                        strength_new.input_ind,
                        t_0=self.t_0,
                        load=load,
                        p_h=TrajectInfo["Pmax"],
                        p_dh=0.5,
                        year=year,
                    )

                    Z, self.p_dh, self.p_dh_c = fds_mechanisms.calculate_z_piping(
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
                    self.scenario_result["beta_cs_p"][j] = calc_beta_implicated(
                        "Piping", self.SF_p * self.gamma_pip, TrajectInfo=TrajectInfo
                    )  #
                    # Calculate the implicated beta_cs

                    # Heave:
                    Z, self.h_i, self.h_i_c = fds_mechanisms.calculate_z_heave(
                        inputs, mode="SemiProb"
                    )
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
                    self.scenario_result["beta_cs_h"][j] = calc_beta_implicated(
                        "Heave",
                        (self.h_i_c / self.gamma_schem_heave) / self.h_i,
                        TrajectInfo=TrajectInfo,
                    )  # Calculate the implicated beta_cs

                    # Uplift
                    Z, self.u_dh, self.u_dh_c = fds_mechanisms.calculate_z_uplift(
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
                    self.scenario_result["beta_cs_u"][j] = calc_beta_implicated(
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
                            self.scenario_result["Pf"][j] = np.max(
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
