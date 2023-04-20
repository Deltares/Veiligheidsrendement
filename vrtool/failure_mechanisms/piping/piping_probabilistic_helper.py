import numpy as np

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta

import logging


class PipingProbabilisticHelper:
    """Class containing functions to calculate probabilistic properties for piping."""

    def __init__(self, traject_info: DikeTrajectInfo) -> None:
        self.traject_info = traject_info

    def calculate_gamma(self, mechanism_name: str) -> np.ndarray:
        """Calculates the gamma based on the mechanism

        Args:
            mechanism_name (str): The name of the mechanism to calculate the gamma for.

        Raises:
            ValueError: Raised when the mechanism is not supported.

        Returns:
            np.ndarray: An array containing the gamma value.
        """
        Pcs = (
            self.traject_info.Pmax
            * self.traject_info.omegaPiping
            * self.traject_info.bPiping
        ) / (self.traject_info.aPiping * self.traject_info.TrajectLength)
        betacs = pf_to_beta(Pcs)
        betamax = self.traject_info.beta_max

        if mechanism_name == "Piping":
            return 1.04 * np.exp(0.37 * betacs - 0.43 * betamax)
        elif mechanism_name == "Heave":
            return 0.37 * np.exp(0.48 * betacs - 0.3 * betamax)
        elif mechanism_name == "Uplift":
            return 0.48 * np.exp(0.46 * betacs - 0.27 * betamax)
        else:
            raise ValueError(f'Mechanism "{mechanism_name}" is not supported.')

    def calculate_implicated_beta(
        self, mechanism_name: str, safety_factor: float
    ) -> np.ndarray:
        """Calculates the implicated reliability from the safety factor.

        Args:
            mechanism_name (str): The name of the mechanism to calculate the reliability for.
            safety_factor (float): The safety factor to calculate the reliabity with.

        Raises:
            ValueError: Raised when the mechanism is not supported.

        Returns:
            np.ndarray: An array containing the implicated reliability.
        """
        if mechanism_name not in ["Piping", "Heave", "Uplift"]:
            raise ValueError(f'Mechanism "{mechanism_name}" is not supported.')

        if safety_factor == 0:
            logging.warn(f'SF for "{mechanism_name}" is 0')
            return 0.5
        elif safety_factor == np.inf:
            return 8

        beta_max = self.traject_info.beta_max
        if mechanism_name == "Piping":
            return (1 / 0.37) * (np.log(safety_factor / 1.04) + 0.43 * beta_max)
        elif mechanism_name == "Heave":
            # TODO troubleshoot the RuntimeWarning errors with invalid values in log.
            return (1 / 0.48) * (np.log(safety_factor / 0.37) + 0.30 * beta_max)
        elif mechanism_name == "Uplift":
            return (1 / 0.46) * (np.log(safety_factor / 0.48) + 0.27 * beta_max)
