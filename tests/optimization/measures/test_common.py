import pytest

from vrtool.optimization.measures.common import floats_are_equal_or_nan


class TestFloatsAreEqualOrNan:
    def test_given_nan_values_returns_true(self):
        assert floats_are_equal_or_nan(float("nan"), float("nan")) is True

    def test_given_simple_float_comparison_returns_true(self):
        assert floats_are_equal_or_nan(4.24, 4.24) is True

    @pytest.mark.parametrize("right_float", [(4.24 + 1.0e-09), (float("nan"))])
    def test_given_different_floats_returns_false(self, right_float: float):
        assert floats_are_equal_or_nan(4.24, right_float) is False
