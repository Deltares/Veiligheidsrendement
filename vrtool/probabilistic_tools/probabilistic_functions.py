from typing import Callable, Optional

import numpy as np
from scipy.stats import norm


def beta_to_pf(beta: float | list[float]) -> float | list[float]:
    # alternative: use scipy
    if isinstance(beta, list):
        return norm.cdf([-_element for _element in beta]).tolist()
    return norm.cdf(-beta)


def pf_to_beta(pf):
    # alternative: use scipy
    return -norm.ppf(pf)


def interpolator(
    xs: list[float],
    ys: list[float],
    left: Optional[float] = None,
    right: Optional[float] = None,
) -> Callable[[float | list[float]], float | list[float]]:
    # Sort values
    sorted_indices = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in sorted_indices]
    ys = [ys[i] for i in sorted_indices]

    # Define slopes for extrapolation
    _left_slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    _right_slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])

    def _interpolator(x: float | list[float]) -> float | list[float]:
        _xs0, _xsN = xs[0], xs[-1]
        _ys0, _ysN = ys[0], ys[-1]

        _x_list = x if isinstance(x, list) else [x]
        _results = []

        for _x in _x_list:
            if _x < _xs0:
                if left is None:
                    # Extrapolate
                    _y = _ys0 + (_x - _xs0) * _left_slope
                else:
                    _y = left
            elif _x > _xsN:
                if right is None:
                    # Extrapolate
                    _y = _ysN + (_x - _xsN) * _right_slope
                else:
                    _y = right
            else:
                _y = float(np.interp(_x, xs, ys))

            _results.append(_y)

        return _results if isinstance(x, list) else _results[0]

    return _interpolator


def interpolate(
    x: float | list[float],
    xs: list[float],
    ys: list[float],
    left: Optional[float] = None,
    right: Optional[float] = None,
) -> float | list[float]:
    _interpolator = interpolator(xs, ys, left=left, right=right)
    return _interpolator(x)
