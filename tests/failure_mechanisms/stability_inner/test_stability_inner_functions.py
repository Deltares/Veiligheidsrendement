import numpy as np
import pytest

from vrtool.failure_mechanisms.stability_inner.stability_inner_functions import (
    calculate_reliability,
    calculate_safety_factor,
)


class TestStabilityInnerFunctions:
    @pytest.mark.parametrize(
        "safety_factor, expected_reliability",
        [
            (1.71, 8),
            (0.0, -2.73),
            (0.5, 0.41),
        ],
    )
    def test_calculate_reliability(self, safety_factor, expected_reliability):
        calculated_reliability = calculate_reliability(
            np.array([safety_factor], dtype=float)
        )

        assert calculated_reliability == pytest.approx(expected_reliability, abs=1e-2)

    @pytest.mark.parametrize(
        "reliability, expected_safety_factor",
        [
            (0, 0.4346),
            (0.5, 0.5141),
            (-0.5, 0.3550),
        ],
    )
    def test_calculate_safety_factory(self, reliability, expected_safety_factor):
        calculated_safety_factor = calculate_safety_factor(reliability)

        assert calculated_safety_factor == pytest.approx(
            expected_safety_factor, abs=1e-4
        )
