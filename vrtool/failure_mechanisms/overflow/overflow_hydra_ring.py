import pandas as pd
import numpy as np
from scipy import interpolate

from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf
from vrtool.failure_mechanisms.overflow.overflow_hydra_ring_input import (
    OverflowHydraRingInput,
)


class OverflowHydraRing:
    def calculate(
        mechanism_input: OverflowHydraRingInput, year: int, initial_year: int
    ) -> tuple[float, float]:
        """
        Calculates the reliability and the probability of failure.

        Args:
            mechanism_input (OverflowHydraRingInput): The input to perform the calculation with.
            year (int): The year to calculate the reliability and the probability of failure for.
            initial_year (int): The initial year.

        Returns:
            tuple[float, float]: A tuple with the calculated reliability and the probability of failure.
        """

        input = dict(
            h_crest=mechanism_input.h_crest,
            d_crest=mechanism_input.d_crest,
            hc_beta=mechanism_input.hc_beta,
        )

        return OverflowHydraRing._calculate_overflow_hydra_ring_assessment(
            input, year, initial_year
        )

    def _calculate_overflow_hydra_ring_assessment(
        self,
        year: int,
        initial_year: int,
        h_crest: float,
        d_crest: float,
        hc_beta: pd.DataFrame,
    ):
        """
        Calculates the overflow based on a HydraRing assessment calculation.
        Args:
            year (int): The year with respect to the starting year to perform the calculation for.
            initial_year (int): The starting year of the calculation.
            h_crest (float): The height of the crest at the initial year.
            d_crest (float): The height correction of the crest per year.
            hc_beta (DataFrame): The hc beta.
        Returns:
            Tuple[float, float]: A tuple with the reliability and the probability of failure.
        """

        h_t = h_crest - d_crest * (year)
        years = hc_beta.columns.values.astype(np.int32)
        betas = []
        for j in years:
            betas.append(
                interpolate.interp1d(
                    hc_beta.index.values,
                    hc_beta[str(j)],
                    fill_value="extrapolate",
                )(h_t)
            )
        beta = interpolate.interp1d(years, betas, fill_value="extrapolate")(
            year + initial_year
        )
        return beta, beta_to_pf(beta)
