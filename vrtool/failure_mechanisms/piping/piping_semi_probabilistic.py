import copy

import numpy as np

from vrtool.probabilistic_tools.probabilistic_functions import (
    add_load_char_vals,
    beta_to_pf,
    calc_beta_implicated,
    pf_to_beta,
)

from vrtool.failure_mechanisms.piping.piping_functions import (
    calculate_z_heave,
    calculate_z_piping,
    calculate_z_uplift,
)

from vrtool.flood_defence_system.mechanism_input import MechanismInput
from vrtool.flood_defence_system.load_input import LoadInput


class PipingSemiProbabilistic:
    def calculate(
        traject_info: dict,
        strength: MechanismInput,
        load: LoadInput,
        year: float,
        t_0: int,
    ) -> tuple[float, float]:
        gamma_schem_heave = 1  # 1.05
        gamma_schem_upl = 1  # 1.05
        gamma_schem_pip = 1  # 1.05

        if traject_info is None:  # Defaults, typical values for 16-3 and 16-4
            traject_info = {}
            traject_info["Pmax"] = 1.0 / 10000
            traject_info["omegaPiping"] = 0.24
            traject_info["bPiping"] = 300
            traject_info["aPiping"] = 0.9
            traject_info["TrajectLength"] = 20000

        # First calculate the SF without gamma for the three submechanisms
        # Piping:
        strength_new = copy.deepcopy(strength)
        scenario_result = {}
        scenario_result["Scenario"] = strength_new.input["Scenario"]
        scenario_result["P_scenario"] = strength_new.input["P_scenario"]
        scenario_result["beta_cs_p"] = {}
        scenario_result["beta_cs_h"] = {}
        scenario_result["beta_cs_u"] = {}
        scenario_result["Pf"] = {}
        scenario_result["Beta"] = {}

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
                t_0=t_0,
                load=load,
                p_h=traject_info["Pmax"],
                p_dh=0.5,
                year=year,
            )

            Z, p_dh, p_dh_c = calculate_z_piping(inputs, mode="SemiProb")
            gamma_pip = traject_info["gammaPiping"]
            # ProbabilisticFunctions.calc_gamma('Piping', TrajectInfo=TrajectInfo) #
            # Calculate needed safety factor

            SF_p = np.inf
            if p_dh != 0:
                SF_p = (p_dh_c / (gamma_pip * gamma_schem_pip)) / p_dh

            assess_p = "voldoende" if SF_p > 1 else "onvoldoende"
            scenario_result["beta_cs_p"][j] = calc_beta_implicated(
                "Piping", SF_p * gamma_pip, traject_info=traject_info
            )  #
            # Calculate the implicated beta_cs

            # Heave:
            Z, h_i, h_i_c = calculate_z_heave(inputs, mode="SemiProb")
            gamma_h = traject_info[
                "gammaHeave"
            ]  # ProbabilisticFunctions.calc_gamma('Heave',TrajectInfo=TrajectInfo)  #
            # Calculate
            # needed safety factor
            # TODO: check formula Sander Kapinga
            SF_h = (h_i_c / (gamma_schem_heave * gamma_h)) / h_i
            assess_h = (
                "voldoende"
                if (h_i_c / (gamma_schem_heave * gamma_h)) / h_i > 1
                else "onvoldoende"
            )
            scenario_result["beta_cs_h"][j] = calc_beta_implicated(
                "Heave",
                (h_i_c / gamma_schem_heave) / h_i,
                traject_info=traject_info,
            )  # Calculate the implicated beta_cs

            # Uplift
            Z, u_dh, u_dh_c = calculate_z_uplift(inputs, mode="SemiProb")
            gamma_u = traject_info[
                "gammaUplift"
            ]  # ProbabilisticFunctions.calc_gamma('Uplift',TrajectInfo=TrajectInfo)
            # Calculate
            # needed safety factor
            # TODO: check formula Sander Kapinga
            SF_u = (u_dh_c / (gamma_schem_upl * gamma_u)) / u_dh

            assess_u = (
                "voldoende"
                if (u_dh_c / (gamma_schem_upl * gamma_u)) / u_dh > 1
                else "onvoldoende"
            )
            scenario_result["beta_cs_u"][j] = calc_beta_implicated(
                "Uplift",
                (u_dh_c / gamma_schem_upl) / u_dh,
                traject_info=traject_info,
            )  # Calculate the implicated beta_cs

            # Check if there is an elimination measure present (VZG or diaphragm wall)
            if "Elimination" in strength.input.keys():
                if strength.input["Elimination"] == "yes":
                    # Fault tree: Pf = P(f|elimination fails)*P(elimination fails) + P(f|elimination works)* P(elimination works)
                    scenario_beta = np.max(
                        [
                            scenario_result["beta_cs_h"][j],
                            scenario_result["beta_cs_u"][j],
                            scenario_result["beta_cs_p"][j],
                        ]
                    )
                    scenario_result["Pf"][j] = np.max(
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
                    scenario_result["Beta"][j] = np.min(
                        [pf_to_beta(scenario_result["Pf"][j]), 8.0]
                    )

                else:
                    raise ValueError("Warning: Elimination defined but not turned on")
            else:
                scenario_result["Beta"][j] = np.min(
                    [
                        np.max(
                            [
                                scenario_result["beta_cs_h"][j],
                                scenario_result["beta_cs_u"][j],
                                scenario_result["beta_cs_p"][j],
                            ]
                        ),
                        8,
                    ]
                )
                scenario_result["Pf"][j] = beta_to_pf(scenario_result["Beta"][j])

        # multiply every scenario by probability
        failure_probability = np.max(
            [
                sum(
                    scenario_result["Pf"][k] * scenario_result["P_scenario"][k]
                    for k in scenario_result["Pf"]
                ),
                beta_to_pf(8.0),
            ]
        )
        beta = np.min([pf_to_beta(failure_probability), 8])

        return [beta, failure_probability]
