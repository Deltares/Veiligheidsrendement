import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

from vrtool.probabilistic_tools.probabilistic_functions import (
    beta_to_pf,
    interpolate,
    pf_to_beta,
)


def calculate_overflow_hydra_ring_design(
    input_dict: dict, year: int, start_year: int, failure_probability: float
) -> tuple[float, float]:
    """
    Calculates the overflow based on a HydraRing design calculation.
    Args:
        input (dict): The input to calculate the overflow with
        year (int): The year with respect to the starting year to perform the calculation for.
        start_year (int): The starting year of the calculation.
        failure_probability (float): The failure probability Pt.
    Returns:
        Tuple[float, float]: A tuple with the calculated height of the new crest and the reliability.
    """

    t_beta_interp = RectBivariateSpline(
        input_dict["hc_beta"].columns.values.astype(np.float32),
        input_dict["hc_beta"].index.values,
        input_dict["hc_beta"],
    )
    h_grid = np.linspace(
        input_dict["hc_beta"].index.values.min(),
        input_dict["hc_beta"].index.values.max(),
        50,
    )
    h_beta = t_beta_interp(year + start_year, h_grid).flatten()
    new_crest = interpolate(pf_to_beta(failure_probability), h_beta, h_grid)

    # add expected crest decline
    new_crest += year * input_dict["d_crest"]

    return new_crest, pf_to_beta(failure_probability)


def calculate_overflow_hydra_ring_assessment(
    year: int, initial_year: int, h_crest: float, d_crest: float, hc_beta: pd.DataFrame
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
    years = hc_beta.columns.values.astype(np.int32).tolist()
    betas = []
    for j in years:
        betas.append(
            interpolate(
                h_t,
                hc_beta.index.values,
                hc_beta[str(j)],
            )
        )
    beta = interpolate(year + initial_year, years, betas)
    return beta, beta_to_pf(beta)


def calculate_overflow_simple_assessment(
    h_crest: np.ndarray,
    q_crest: np.ndarray,
    h_c: np.ndarray,
    q_c: np.ndarray,
    beta: np.ndarray,
):
    """
    Calculates the overflow with a simple approximation of the assessment calculation.
    Args:
        h_crest (ndarray): Current creat height.
        q_crest (ndarray): Critical crest height.
        h_c (ndarray): _description_
        q_c (ndarray): _description_
        beta (ndarray): The reliability
    Returns:
        Tuple[float, float]: A tuple with the reliability and the probability of failure.
    """

    if q_c[0] != q_c[-1:]:
        beta_hc = RectBivariateSpline(h_c, q_c, beta)
        beta = np.min([beta_hc(h_crest, q_crest), 8.0])
    else:
        beta_hc = interpolate(h_crest.tolist(), h_c.tolist(), beta.tolist())
        beta = np.min([beta_hc, [8.0]])

    return beta, beta_to_pf(beta)


def calculate_overflow_simple_design(
    q_crest: np.ndarray,
    h_c: np.ndarray,
    q_c: np.ndarray,
    beta: np.ndarray,
    failure_probability: float,
    design_variable: str,
):
    """
    Calculates the overflow with a simple approximation for the design calculation.
    Args:
        q_crest (ndarray): Critical crest height.
        h_c (ndarray): _description_
        q_c (ndarray): _description_
        beta (ndarray): The reliability
        failure_probability (float, optional): The failure probability Pt. Defaults to None.
        design_variable (str, optional): The design variable to calculate for. Defaults to None.
    Returns:
        Tuple[float, float]: A tuple with the calculated height of the new crest and the reliability.
    """

    beta_t = pf_to_beta(failure_probability)
    if design_variable == "h_crest":
        if q_c[0] != q_c[-1:]:
            beta_hc = interpolate.RectBivariateSpline(beta, q_c, h_c)
            h_crest = beta_hc(beta_t, q_crest)
        else:
            beta_hc = interpolate(beta, h_c, kind="linear", fill_value="extrapolate")
            h_crest = beta_hc(beta_t)

        return h_crest, beta_t
