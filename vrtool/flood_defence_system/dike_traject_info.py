from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


@dataclass
class DikeTrajectInfo:
    "Class containing relevant failure mechanism information for a dike traject."
    traject_name: str

    omegaPiping: float = 0.24
    omegaStabilityInner: float = 0.04
    omegaOverflow: float = 0.24

    aPiping: float = float("nan")
    bPiping: float = 300

    aStabilityInner: float = 0.033
    bStabilityInner: float = 50

    beta_max: float = float("nan")
    Pmax: float = float("nan")
    FloodDamage: float = float("nan")
    TrajectLength: float = 0

    def calculate_gamma(self, mechanism: str) -> np.ndarray:
        """Calculates the gamma based on the mechanism

        Args:
            mechanism (str): The mechanism to calculate the gamma for.

        Raises:
            NotImplementedError: Raised when the mechanism is not supported.

        Returns:
            np.ndarray: An array containing the gamma value.
        """
        Pcs = (self.Pmax * self.omegaPiping * self.bPiping) / (
            self.aPiping * self.TrajectLength
        )
        betacs = pf_to_beta(Pcs)
        betamax = pf_to_beta(self.Pmax)
        
        if mechanism == "Piping":
            return 1.04 * np.exp(0.37 * betacs - 0.43 * betamax)
        elif mechanism == "Heave":
            return 0.37 * np.exp(0.48 * betacs - 0.3 * betamax)
        elif mechanism == "Uplift":
            return 0.48 * np.exp(0.46 * betacs - 0.27 * betamax)
        else:
            raise NotImplementedError("Mechanism not found")

    # Function to calculate the implicated reliability from the safety factor
    def calc_beta_implicated(self, mechanism: str, safety_factor: float) -> np.ndarray:
        if safety_factor == 0:
            logging.warn("SF for " + mechanism + " is 0")
            beta = 0.5
        elif safety_factor == np.inf:
            beta = 8
        else:
            if mechanism == "Piping":
                beta = (1 / 0.37) * (
                    np.log(safety_factor / 1.04) + 0.43 * self.beta_max
                )
            elif mechanism == "Heave":
                # TODO troubleshoot the RuntimeWarning errors with invalid values in log.
                beta = (1 / 0.48) * (
                    np.log(safety_factor / 0.37) + 0.30 * self.beta_max
                )
            elif mechanism == "Uplift":
                beta = (1 / 0.46) * (
                    np.log(safety_factor / 0.48) + 0.27 * self.beta_max
                )
            else:
                logging.warn("Mechanism not found")
        return beta

    def __post_init__(self):
        self.beta_max = pf_to_beta(self.Pmax)
        self.gammaHeave = self.calculate_gamma("Heave")
        self.gammaUplift = self.calculate_gamma("Uplift")
        self.gammaPiping = self.calculate_gamma("Piping")

    @classmethod
    def from_traject_name(cls, traject_name: str) -> DikeTrajectInfo:
        # Basic traject info
        # Flood damage is based on Economic damage in 2011 as given in https://www.helpdeskwater.nl/publish/pages/132790/factsheets_compleet19122016.pdf
        # Pmax is the ondergrens as given by law
        # TODO check whether this is a sensible value
        # TODO read these values from a generic input file.
        _general_info = {"traject_name": traject_name}
        if traject_name == "16-4":
            _general_info["aPiping"] = 0.9
            _general_info["FloodDamage"] = 23e9
            _general_info["TrajectLength"] = 19480
            _general_info["Pmax"] = 1.0 / 10000
        elif traject_name == "16-3":
            _general_info["aPiping"] = 0.9
            _general_info["FloodDamage"] = 23e9
            _general_info["TrajectLength"] = 19899
            _general_info["Pmax"] = 1.0 / 10000
            # NB: klopt a hier?????!!!!
        elif traject_name == "16-3 en 16-4":
            _general_info["aPiping"] = 0.9
            _general_info["FloodDamage"] = 23e9
            # voor doorsnede-eisen wel ongeveer lengte individueel traject
            _general_info["TrajectLength"] = 19500
            # gebruiken
            _general_info["Pmax"] = 1.0 / 10000
        elif traject_name == "38-1":
            _general_info["aPiping"] = 0.9
            _general_info["FloodDamage"] = 14e9
            # voor doorsnede-eisen wel ongeveer lengte individueel traject
            _general_info["TrajectLength"] = 29500
            # gebruiken
            _general_info["Pmax"] = 1.0 / 30000
        elif traject_name == "16-1":
            _general_info["aPiping"] = 0.4
            _general_info["FloodDamage"] = 29e9
            _general_info["TrajectLength"] = 15000
            _general_info["Pmax"] = 1.0 / 30000
        else:
            logging.warn(
                "Dike traject not found, using default assumptions for traject."
            )
            _general_info["aPiping"] = 0.9
            _general_info["FloodDamage"] = 5e9
            _general_info["Pmax"] = 1.0 / 10000
            _general_info["omegaPiping"] = 0.24
            _general_info["omegaStabilityInner"] = 0.04
            _general_info["omegaOverflow"] = 0.24
            _general_info["bPiping"] = 300
            _general_info["aStabilityInner"] = 0.033
            _general_info["bStabilityInner"] = 50

        return cls(**_general_info)
