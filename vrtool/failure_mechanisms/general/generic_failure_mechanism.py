import numpy as np
from scipy import interpolate

from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf
from vrtool.failure_mechanisms.general.generic_failure_mechanism_input import (
    GenericFailureMechanismInput,
)


class GenericFailureMechanism:
    def calculate(
        mechanism_input: GenericFailureMechanismInput, year: int
    ) -> tuple[float, float]:
        """
        Calculates the reliability and safety factor based on its input arguments.

        Args:
            mechanism_input (DirectFailureMechanismInput): The relevant input for this failure mechanism.
            year (int): The year to calculate the reliability and safety factor for.

        Returns:
            Tuple[float, float]: A tuple containing the reliability and safety factor.
        """

        betat = interpolate.interp1d(
            mechanism_input.time_grid,
            mechanism_input.beta_grid,
            fill_value="extrapolate",
        )
        beta = np.float32(betat(year))

        return [beta, beta_to_pf(beta)]
