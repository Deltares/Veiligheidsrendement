from typing import Tuple, list

import numpy as np
from scipy import interpolate

from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


def calculate_reliability(
    t_grid: list[float], beta_grid: list[float], year: int
) -> Tuple[float, float]:
    """
    Calculates the reliability and safety factor based on its input arguments.

    Args:
        t_grid (list[float]): The list of years for each reliability.
        beta_grid (list[float]): The reliability associated with each year in t_grid.
        year (int): The year to calculate the reliability and safety factor for.

    Returns:
        Tuple[float, float]: A tuple containing the reliability and safety factor.
    """

    betat = interpolate.interp1d(t_grid, beta_grid, fill_value="extrapolate")
    beta = np.float32(betat(year))

    return [beta, beta_to_pf(beta)]
