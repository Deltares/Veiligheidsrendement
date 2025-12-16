import pytest

from vrtool.probabilistic_tools.probabilistic_functions import interpolate, interpolator


class TestProbabilisticFunctions:
    @pytest.mark.parametrize(
        "x, expected",
        [
            pytest.param(0.5, 5.0, id="single value"),
            pytest.param([0.0, 0.5, 1.0], [0.0, 5.0, 10.0], id="list values"),
            pytest.param(-1.0, -10.0, id="extrapolate left"),
            pytest.param(3.0, 30.0, id="extrapolate right"),
        ],
    )
    def test_interpolator(self, x: float | list[float], expected: float | list[float]):
        # 1. Define test data.
        _xs = [0.0, 1.0, 2.0]
        _ys = [0.0, 10.0, 20.0]
        _interpolator = interpolator(_xs, _ys)

        # 2. Execute test.
        _result = _interpolator(x)

        # 3. Verify results.
        assert _result == expected

    def test_interpolator_unsorted_data(self):
        # 1. Define test data.
        _xs = [2.0, 0.0, 1.0]
        _ys = [20.0, 0.0, 10.0]
        _interpolator = interpolator(_xs, _ys)

        # 2. Execute test.
        x = 0.5
        expected = 5.0
        _result = _interpolator(x)

        # 3. Verify results.
        assert _result == expected

    @pytest.mark.parametrize(
        "x, expected",
        [
            pytest.param(0.5, 5.0, id="single value"),
            pytest.param([0.0, 0.5, 1.0], [0.0, 5.0, 10.0], id="list values"),
            pytest.param(-1.0, -10.0, id="extrapolate left"),
            pytest.param(3.0, 30.0, id="extrapolate right"),
        ],
    )
    def test_interpolate(self, x: float | list[float], expected: float | list[float]):
        # 1. Define test data.
        _xs = [0.0, 1.0, 2.0]
        _ys = [0.0, 10.0, 20.0]

        # 2. Execute test.
        _result = interpolate(x, _xs, _ys)

        # 3. Verify results.
        assert _result == expected

    @pytest.mark.parametrize(
        "x, expected",
        [
            pytest.param(-1.0, -5.0, id="beyond left bound"),
            pytest.param(3.0, 25.0, id="beyond right bound"),
            pytest.param([-1.0, 3.0], [-5.0, 25.0], id="both sides beyond bounds"),
            pytest.param(0.5, 5.0, id="within bounds"),
            pytest.param([0.0, 0.5, 1.0], [0.0, 5.0, 10.0], id="list within bounds"),
        ],
    )
    def test_interpolate_with_bounds(
        self, x: float | list[float], expected: float | list[float]
    ):
        # 1. Define test data.
        _xs = [0.0, 1.0, 2.0]
        _ys = [0.0, 10.0, 20.0]
        _left_bound = -5.0
        _right_bound = 25.0

        # 2. Execute test with left and right bounds.
        _result = interpolate(x, _xs, _ys, left=_left_bound, right=_right_bound)

        # 3. Verify results.
        assert _result == expected
