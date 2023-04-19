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

    def _calculate_gamma(self, mechanism: str) -> np.ndarray:
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

    def __post_init__(self):
        self.beta_max = pf_to_beta(self.Pmax)
        self.gammaHeave = self.__calculate_gamma("Heave")
        self.gammaUplift = self.__calculate_gamma("Uplift")
        self.gammaPiping = self.__calculate_gamma("Piping")

    @classmethod
    def from_traject_info(
        cls, traject_name: str, traject_length: float
    ) -> DikeTrajectInfo:
        # Basic traject info
        # Flood damage is based on Economic damage in 2011 as given in https://www.helpdeskwater.nl/publish/pages/132790/factsheets_compleet19122016.pdf
        # Pmax is the ondergrens as given by law
        # TODO check whether this is a sensible value
        # TODO read these values from a generic input file.
        _general_info = {"traject_name": traject_name}
        if traject_name in ["16-4", "16-3", "16-3 en 16-4"]:
            _general_info["aPiping"] = 0.9
            _general_info["FloodDamage"] = 23e9
            _general_info["TrajectLength"] = traject_length
            _general_info["Pmax"] = 1.0 / 10000
        elif traject_name == "38-1":
            _general_info["aPiping"] = 0.9
            _general_info["FloodDamage"] = 14e9
            # voor doorsnede-eisen wel ongeveer lengte individueel traject
            _general_info["TrajectLength"] = traject_length
            # gebruiken
            _general_info["Pmax"] = 1.0 / 30000
        elif traject_name == "16-1":
            _general_info["aPiping"] = 0.4
            _general_info["FloodDamage"] = 29e9
            _general_info["TrajectLength"] = traject_length
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
            _general_info["TrajectLength"] = traject_length

        return cls(**_general_info)
