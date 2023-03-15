import numpy as np
import pandas as pd
from scipy import interpolate

from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


def calculate_overflow_hydra_ring_design(
    input: dict, year: int, start_year: int, failure_probability: float
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

    t_beta_interp = interpolate.interp2d(
        input["hc_beta"].columns.values.astype(np.float32),
        input["hc_beta"].index.values,
        input["hc_beta"],
        bounds_error=False,
    )
    h_grid = np.linspace(
        input["hc_beta"].index.values.min(),
        input["hc_beta"].index.values.max(),
        50,
    )
    h_beta = t_beta_interp(year + start_year, h_grid).flatten()
    new_crest = interpolate.interp1d(h_beta, h_grid, fill_value="extrapolate")(
        pf_to_beta(failure_probability)
    ).item()

    # add expected crest decline
    new_crest += year * input["d_crest"]

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
        beta_hc = interpolate.interp2d(
            h_c, q_c, beta, kind="linear", fill_value="extrapolate"
        )
        beta = np.min([beta_hc(h_crest, q_crest), 8.0])
    else:
        beta_hc = interpolate.interp1d(
            h_c, beta, kind="linear", fill_value="extrapolate"
        )
        beta = np.min([beta_hc(h_crest), [8.0]])

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
            beta_hc = interpolate.interp2d(
                beta, q_c, h_c, kind="linear", fill_value="extrapolate"
            )
            h_crest = beta_hc(beta_t, q_crest)
        else:
            beta_hc = interpolate.interp1d(
                beta, h_c, kind="linear", fill_value="extrapolate"
            )
            h_crest = beta_hc(beta_t)

        return h_crest, beta_t
