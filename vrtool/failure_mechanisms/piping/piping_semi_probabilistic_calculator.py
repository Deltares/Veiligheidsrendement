import copy

import numpy as np

from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.piping.piping_functions import (
    calculate_z_heave,
    calculate_z_piping,
    calculate_z_uplift,
)
from vrtool.failure_mechanisms.piping.piping_probabilistic_helper import (
    PipingProbabilisticHelper,
)
from vrtool.failure_mechanisms.piping.piping_failure_submechanism import (
    PipingFailureSubmechanism,
)
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.probabilistic_tools.probabilistic_functions import (
    add_load_char_vals,
    beta_to_pf,
    pf_to_beta,
)
from vrtool.common.dike_traject_info import DikeTrajectInfo


class PipingSemiProbabilisticCalculator(FailureMechanismCalculatorProtocol):
    def __init__(
        self,
        mechanism_input: MechanismInput,
        load: LoadInput,
        initial_year: int,
        traject_info: DikeTrajectInfo,
    ) -> None:
        """Initializes a calculator for piping semi-probabilistic calculations

        Args:
            mechanism_input (MechanismInput): The input for the mechanism.
            load (LoadInput): The load input.
            initial_year (int): The initial year to base the calculation on.
            traject_info (DikeTrajectInfo): An object containing the traject info.
        """
        if not isinstance(mechanism_input, MechanismInput):
            raise ValueError(
                "Expected instance of a {}.".format(MechanismInput.__name__)
            )

        if not isinstance(load, LoadInput):
            raise ValueError("Expected instance of a {}.".format(LoadInput.__name__))

        if not isinstance(initial_year, int):
            raise ValueError("Expected instance of a {}.".format(int.__name__))

        if not isinstance(traject_info, DikeTrajectInfo):
            raise ValueError(
                "Expected instance of a {}.".format(DikeTrajectInfo.__name__)
            )

        self._mechanism_input = mechanism_input
        self._load = load
        self._traject_info = traject_info
        self._initial_year = initial_year
        self._probabilistic_helper = PipingProbabilisticHelper(traject_info)

    def calculate(self, year: float) -> tuple[float, float]:
        # First calculate the SF without gamma for the three submechanisms
        # Piping:
        strength_new = copy.deepcopy(self._mechanism_input)
        scenario_result = {}
        scenario_result["Scenario"] = strength_new.input["Scenario"]
        scenario_result["P_scenario"] = strength_new.input["P_scenario"]
        scenario_result["beta_cs_p"] = {}
        scenario_result["beta_cs_h"] = {}
        scenario_result["beta_cs_u"] = {}
        scenario_result["Pf"] = {}
        scenario_result["Beta"] = {}

        for i in self._mechanism_input.temporals:
            strength_new.input[i] = self._mechanism_input.input[i] * year

        # TODO:below, remove self. in for example self.gamma_pip. This is just an scenario output value. do not store.
        # calculate beta per scenario and determine overall
        for scenario in range(0, len(strength_new.input["Scenario"])):
            strength_new.input_ind = {}
            for i in strength_new.input:  # select values of scenario j
                try:
                    strength_new.input_ind[i] = strength_new.input[i][scenario]
                except:
                    pass  # TODO: make more clean, na measures doorloopt hij deze loop nogmaals, niet voor alle variabelen in strength_new.input is een array beschikbaar.

            # inputs = addLoadCharVals(strength_new.input, load=None, p_h=TrajectInfo['Pmax'], p_dh=0.5, year=year)
            # inputs['h'] = load.NormWaterLevel
            # TODO aanpassen met nieuwe belastingmodel
            inputs = add_load_char_vals(
                strength_new.input_ind,
                t_0=self._initial_year,
                load=self._load,
                p_h=self._traject_info.Pmax,
                p_dh=0.5,
                year=year,
            )

            # Piping
            scenario_result["beta_cs_p"][scenario] = self._calculate_beta_piping(inputs)

            # Heave:
            scenario_result["beta_cs_h"][scenario] = self._calculate_beta_heave(inputs)

            # Uplift
            scenario_result["beta_cs_u"][scenario] = self._calculate_beta_uplift(inputs)

            # Check if there is an elimination measure present (VZG or diaphragm wall)
            if "Elimination" in self._mechanism_input.input.keys():
                if self._mechanism_input.input["Elimination"] == "yes":
                    # Fault tree: Pf = P(f|elimination fails)*P(elimination fails) + P(f|elimination works)* P(elimination works)
                    scenario_beta = np.max(
                        [
                            scenario_result["beta_cs_h"][scenario],
                            scenario_result["beta_cs_u"][scenario],
                            scenario_result["beta_cs_p"][scenario],
                        ]
                    )
                    scenario_result["Pf"][scenario] = np.max(
                        [
                            np.min(
                                [
                                    beta_to_pf(scenario_beta)
                                    * self._mechanism_input.input["Pf_elim"]
                                    + self._mechanism_input.input["Pf_with_elim"]
                                    * (1 - self._mechanism_input.input["Pf_elim"]),
                                    beta_to_pf(scenario_beta),
                                ]
                            ),
                            beta_to_pf(8.0),
                        ]
                    )
                    scenario_result["Beta"][scenario] = np.min(
                        [pf_to_beta(scenario_result["Pf"][scenario]), 8.0]
                    )

                else:
                    raise ValueError("Warning: Elimination defined but not turned on")
            else:
                scenario_result["Beta"][scenario] = np.min(
                    [
                        np.max(
                            [
                                scenario_result["beta_cs_h"][scenario],
                                scenario_result["beta_cs_u"][scenario],
                                scenario_result["beta_cs_p"][scenario],
                            ]
                        ),
                        8,
                    ]
                )
                scenario_result["Pf"][scenario] = beta_to_pf(
                    scenario_result["Beta"][scenario]
                )

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

    def _calculate_beta_piping(self, inputs: dict):
        submechanism = PipingFailureSubmechanism.PIPING
        gamma_schem_pip = 1  # 1.05

        Z, p_dh, p_dh_c = calculate_z_piping(inputs, mode="SemiProb")
        gamma_pip = self._probabilistic_helper.calculate_gamma(submechanism)

        # Calculate needed safety factor
        SF_p = np.inf
        if p_dh != 0:
            SF_p = (p_dh_c / (gamma_pip * gamma_schem_pip)) / p_dh

        return self._probabilistic_helper.calculate_implicated_beta(
            submechanism, SF_p * gamma_pip
        )

    def _calculate_beta_heave(self, inputs: dict):
        submechanism = PipingFailureSubmechanism.HEAVE
        gamma_schem_heave = 1  # 1.05

        Z, h_i, h_i_c = calculate_z_heave(inputs, mode="SemiProb")
        gamma_h = self._probabilistic_helper.calculate_gamma(submechanism)

        # Calculate
        # needed safety factor
        # TODO: check formula Sander Kapinga
        SF_h = (h_i_c / (gamma_schem_heave * gamma_h)) / h_i
        return self._probabilistic_helper.calculate_implicated_beta(
            submechanism, (h_i_c / gamma_schem_heave) / h_i
        )  # Calculate the implicated beta_cs

    def _calculate_beta_uplift(self, inputs: dict):
        submechanism = PipingFailureSubmechanism.UPLIFT
        gamma_schem_upl = 1  # 1.05

        Z, u_dh, u_dh_c = calculate_z_uplift(inputs, mode="SemiProb")
        gamma_u = self._probabilistic_helper.calculate_gamma(submechanism)

        # Calculate
        # needed safety factor
        # TODO: check formula Sander Kapinga
        SF_u = (u_dh_c / (gamma_schem_upl * gamma_u)) / u_dh

        return self._probabilistic_helper.calculate_implicated_beta(
            submechanism, (u_dh_c / gamma_schem_upl) / u_dh
        )  # Calculate the implicated beta_cs
