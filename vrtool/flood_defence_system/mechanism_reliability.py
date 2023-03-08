import copy

import numpy as np
import openturns as ot

import vrtool.flood_defence_system.mechanisms as fds_mechanisms
from vrtool.failure_mechanisms.overflow.overflow import Overflow
from vrtool.flood_defence_system.mechanism_input import MechanismInput
from vrtool.probabilistic_tools.probabilistic_functions import (
    add_load_char_vals,
    beta_to_pf,
    calc_beta_implicated,
    pf_to_beta,
)

from vrtool.failure_mechanisms.stability_inner import (
    StabilityInnerInput,
    StabilityInner,
)
from vrtool.failure_mechanisms.general import (
    GenericFailureMechanismInput,
    GenericFailureMechanism,
)

from vrtool.failure_mechanisms.overflow import OverflowSimpleInput, Overflow


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
            self.beta, self.Pf = self._calculate_direct_input(strength, year)

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
                self.beta, self.Pf = self._calculate_simple_stability_inner(
                    strength, year
                )
            elif mechanism == "Overflow":  # specific for SAFE
                self.beta, self.Pf = self._calculate_simple_overflow(
                    strength, year, load
                )
            elif mechanism == "Piping":
                pass

            self.alpha_sq = np.nan
            self.result = np.nan

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
                        "Piping", self.SF_p * self.gamma_pip, traject_info=TrajectInfo
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
                        traject_info=TrajectInfo,
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
                        traject_info=TrajectInfo,
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

    def _calculate_direct_input(
        self, mechanism_input: MechanismInput, year: int
    ) -> tuple[float, float]:
        _mechanism_input = GenericFailureMechanismInput.from_mechanism_input(
            mechanism_input
        )
        return GenericFailureMechanism.calculate_reliability(_mechanism_input, year)

    def _calculate_simple_stability_inner(
        self, mechanism_input: MechanismInput, year: int
    ) -> tuple[float, float]:
        _mechanism_input = StabilityInnerInput.from_mechanism_input(mechanism_input)
        return StabilityInner.calculate_simple(_mechanism_input, year)

    def _calculate_simple_overflow(
        self, mechanism_input: MechanismInput, year: int, load
    ) -> tuple[float, float]:
        # climate change included, including a factor for HBN
        if hasattr(load, "dist_change"):
            corrected_crest_height = (
                mechanism_input.input["h_crest"]
                - (
                    mechanism_input.input["dhc(t)"]
                    + (load.dist_change * load.HBN_factor)
                )
                * year
            )
        else:
            corrected_crest_height = mechanism_input.input["h_crest"] - (
                mechanism_input.input["dhc(t)"] * year
            )

        _mechanism_input = OverflowSimpleInput.from_mechanism_input(
            mechanism_input, corrected_crest_height
        )
        return Overflow.calculate_simple(_mechanism_input)
