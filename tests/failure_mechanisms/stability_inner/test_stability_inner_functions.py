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
            pytest.param(1.71, 8, id="Above threshold"),
            pytest.param(0.0, -2.73, id="Below threshold (1)"),
            pytest.param(0.5, 0.41, id="Below threshold (2)"),
        ],
    )
    def test_calculate_reliability(
        self, safety_factor: float, expected_reliability: float
    ):
        # Call
        calculated_reliability = calculate_reliability(
            np.array([safety_factor], dtype=float)
        )

        # Assert
        assert calculated_reliability == pytest.approx(expected_reliability, abs=1e-2)

    def test_calculate_reliability_given_values_greater_than_threshold_returns_threshold(
        self,
    ):
        # 1. Define test data.
        _safety_factor_above_threshold = 1.71
        _safety_factor_below_threshold = 0.5
        _safety_factory_array = [
            _safety_factor_above_threshold,
            _safety_factor_below_threshold,
        ]
        _built_in_threshold = 8.0
        _expected_reliability_array = [_built_in_threshold, 0.41]

        # 2. Run test.
        _calculated_reliability = calculate_reliability(
            np.array(_safety_factory_array, dtype=float)
        )

        # Assert
        assert isinstance(_calculated_reliability, np.ndarray)
        for _result, expectation in zip(
            _calculated_reliability, _expected_reliability_array
        ):
            assert _result == pytest.approx(expectation, abs=1e-2)

    @pytest.mark.parametrize(
        "reliability, expected_safety_factor",
        [
            (0, 0.4346),
            (0.5, 0.5141),
            (-0.5, 0.3550),
        ],
    )
    def test_calculate_safety_factor(
        self, reliability: float, expected_safety_factor: float
    ):
        # Call
        calculated_safety_factor = calculate_safety_factor(reliability)

        # Assert
        assert calculated_safety_factor == pytest.approx(
            expected_safety_factor, abs=1e-4
        )
