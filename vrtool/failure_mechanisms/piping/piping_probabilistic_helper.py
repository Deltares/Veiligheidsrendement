import logging

import numpy as np

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.failure_mechanisms.piping.piping_failure_submechanism import (
    PipingFailureSubmechanism,
)
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


class PipingProbabilisticHelper:
    """Helper class containing functions to calculate probabilistic properties for piping."""

    def __init__(self, traject_info: DikeTrajectInfo) -> None:
        """Initializes the helper class containing probabilistic properties for piping.

        Args:
            traject_info (DikeTrajectInfo): An object containing the dike traject info.

        Raises:
            ValueError: Raised when traject_info is invalid.
        """
        if not isinstance(traject_info, DikeTrajectInfo):
            raise ValueError(
                "Expected instance of a {}.".format(DikeTrajectInfo.__name__)
            )

        self.traject_info = traject_info

    def calculate_gamma(self, submechanism: PipingFailureSubmechanism) -> np.ndarray:
        """Calculates the gamma based on the mechanism

        Args:
            submechanism (PipingFailureSubmechanism): The name of the piping failure sub mechanism to calculate the gamma for.

        Raises:
            ValueError: Raised when the mechanism is not supported.

        Returns:
            np.ndarray: An array containing the gamma value.
        """
        PipingFailureSubmechanism.is_valid(submechanism)

        Pcs = (
            self.traject_info.Pmax
            * self.traject_info.omegaPiping
            * self.traject_info.bPiping
        ) / (self.traject_info.aPiping * self.traject_info.TrajectLength)
        betacs = pf_to_beta(Pcs)
        betamax = self.traject_info.beta_max

        match submechanism:
            case PipingFailureSubmechanism.PIPING:
                return 1.04 * np.exp(0.37 * betacs - 0.43 * betamax)
            case PipingFailureSubmechanism.HEAVE:
                return 0.37 * np.exp(0.48 * betacs - 0.3 * betamax)
            case PipingFailureSubmechanism.UPLIFT:
                return 0.48 * np.exp(0.46 * betacs - 0.27 * betamax)
            case _:
                raise ValueError(
                    f"Unsupported value of {PipingFailureSubmechanism.__name__}."
                )

    def calculate_implicated_beta(
        self, submechanism: PipingFailureSubmechanism, safety_factor: float
    ) -> np.ndarray:
        """Calculates the implicated reliability from the safety factor.

        Args:
            sub_mechanism (PipingFailureSubmechanism): The type of the piping failure submechanism to calculate the reliability for.
            safety_factor (float): The safety factor to calculate the reliabity with.

        Raises:
            ValueError: Raised when the mechanism is not supported.

        Returns:
            np.ndarray: An array containing the implicated reliability.
        """
        PipingFailureSubmechanism.is_valid(submechanism)

        if safety_factor == 0:
            logging.debug(f'SF for "{submechanism}" is 0')
            return 0.5
        elif safety_factor == np.inf:
            return 8

        beta_max = self.traject_info.beta_max
        if submechanism == PipingFailureSubmechanism.PIPING:
            return (1 / 0.37) * (np.log(safety_factor / 1.04) + 0.43 * beta_max)
        elif submechanism == PipingFailureSubmechanism.HEAVE:
            # TODO troubleshoot the RuntimeWarning errors with invalid values in log.
            return (1 / 0.48) * (np.log(safety_factor / 0.37) + 0.30 * beta_max)
        elif submechanism == PipingFailureSubmechanism.UPLIFT:
            return (1 / 0.46) * (np.log(safety_factor / 0.48) + 0.27 * beta_max)
