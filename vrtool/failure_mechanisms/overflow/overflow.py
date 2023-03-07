from typing import List, Tuple

import numpy as np
from scipy import interpolate

from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


class Overflow:
    """
    Contains all methods to perform an overflow calculation
    """

    def overflow_hring(
        input,
        year: int,
        start_year: int,
        mode: str = "assessment",
        failure_probability: float = None,
    ) -> Tuple[float, float]:
        """
        Calculates the overflow based on a HydraRing.
        Args:
            input (__type__): The input to calculate the overflow with
            year (int): The year with respect to the starting year to perform the calculation for.
            start_year (int): The starting year of the calculation.
            mode (str, optional): The calculation mode. Defaults to "assessment".
            failure_probability (float, optional): The failure probability Pt. Defaults to None.
        Returns:
            Tuple[float, float]: A tuple with the calculated height of the new crest and the reliability.
        """

        if mode == "assessment":
            h_t = input["h_crest"] - input["d_crest"] * (year)
            years = input["hc_beta"].columns.values.astype(np.int32)
            betas = []
            for j in years:
                betas.append(
                    interpolate.interp1d(
                        input["hc_beta"].index.values,
                        input["hc_beta"][str(j)],
                        fill_value="extrapolate",
                    )(h_t)
                )
            beta = interpolate.interp1d(years, betas, fill_value="extrapolate")(
                year + start_year
            )
            return beta, beta_to_pf(beta)
        if mode == "design":
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
            h_beta = t_beta_interp(year + t_0, h_grid).flatten()
            new_crest = interpolate.interp1d(h_beta, h_grid, fill_value="extrapolate")(
                pf_to_beta(failure_probability)
            ).item()
            return new_crest, pf_to_beta(failure_probability)

    def overflow_simple(
        h_crest: float,
        q_crest: float,
        h_c: float,
        q_c: float,
        beta: float,
        mode: str = "assessment",
        failure_probability: float = None,
        design_variable: str = None,
        iterative_solve: bool = False,
        beta_t: bool = False,
    ) -> Tuple[float, float]:
        """
        Calculates the overflow with a simple approximation.
        Args:
            h_crest (float): Current creat height.
            q_crest (float): Critical crest height.
            h_c (float): _description_
            q_c (float): _description_
            beta (float): The reliability
            mode (str, optional): The calculation mode. Defaults to "assessment".
            failure_probability (float, optional): The failure probability Pt. Defaults to None.
            design_variable (str, optional): The design variable to calculate for. Defaults to None.
            iterative_solve (bool, optional): Flag whether the solution needs to be solved iteratively. Defaults to False.
            beta_t (bool, optional): _description_. Defaults to False.
        Returns:
            Tuple[float, float]: A tuple with the calculated height of the new crest and the reliability.
        """

        if mode == "assessment":
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
            Pf = beta_to_pf(beta)
            if not iterative_solve:
                return beta, Pf
            else:
                return beta - beta_t
        elif mode == "design":
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
            pass
