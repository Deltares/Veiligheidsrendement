import numpy as np
import pytest

from vrtool.probabilistic_tools.table_dist import TableDist


class TestTableDist:
    def test_initialize(self):
        # 1. Define test data.
        _x = np.array([0.0, 1.0, 2.0])
        _p = np.array([0.0, 0.5, 1.0])

        # 2. Run test.
        _dist = TableDist(_x, _p)

        # 3. Verify expectations.
        assert _dist.x[0] == 2e-8
        assert _dist.x[-1] == _x[-1]
        assert _dist.p[0] == _p[0]
        assert _dist.p[-1] == _p[-1]

    def test_initialize_is_load_true(self):
        # 1. Define test data.
        _x = np.array([2.0, 3.0, 5.0])
        _p = np.array([0.1, 0.5, 1.0])
        _gridpoints = 1000

        # 2. Run test.
        _dist = TableDist(_x, _p, extrap=True, isload=True, gridpoints=_gridpoints)

        # 3. Verify expectations.
        assert _dist.x[0] == pytest.approx(1.8433, rel=1e-4)
        assert _dist.x[-1] == pytest.approx(_x[-1], rel=1e-4)
        assert _dist.p[0] == 0.0
        assert _dist.p[-1] == _p[-1]

    @pytest.mark.parametrize(
        "p",
        [
            pytest.param(np.array([0.1, 0.5, 0.9, 1.0]), id="p too long"),
            pytest.param(np.array([0.0, 0.5]), id="p too short"),
        ],
    )
    def test_initialize_unequal_length_raises(self, p: np.ndarray):
        # 1. Define test data.
        _x = np.array([0.0, 1.0, 2.0])

        # 2. Run test.
        with pytest.raises(ValueError) as exc_info:
            _ = TableDist(_x, p)

        # 3. Verify expectations.
        assert "Input arrays have unequal lengths" in str(exc_info.value)

    @pytest.mark.parametrize(
        "p",
        [
            pytest.param(np.array([0.1, 0.5, 1.0]), id="0.0 missing"),
            pytest.param(np.array([0.0, 0.5, 0.9]), id="1.0 missing"),
        ],
    )
    def test_initialize_extrapolate_false_incomplete_p_raises(self, p: np.ndarray):
        # 1. Define test data.
        _x = np.array([0.0, 1.0, 2.0])

        # 2. Run test.
        with pytest.raises(ValueError) as exc_info:
            _ = TableDist(_x, p, extrap=False)

        # 3. Verify expectations.
        assert "Probability bounds are not equal to 0 and 1" in str(exc_info.value)

    @pytest.mark.parametrize(
        "x",
        [
            pytest.param(np.array([2.0, 1.0, 0.0]), id="Decreasing"),
            pytest.param(np.array([1.0, 0.0, 1.0]), id="U-shape"),
            pytest.param(np.array([0.0, 1.0, 0.0]), id="N-shape"),
        ],
    )
    def test_initialize_non_increasing_x_raises(self, x: np.ndarray):
        # 1. Define test data.
        _p = np.array([0.0, 0.5, 1.0])

        # 2. Run test.
        with pytest.raises(ValueError) as exc_info:
            _ = TableDist(x, _p)

        # 3. Verify expectations.
        assert "Values should be increasing" in str(exc_info.value)

    def test_initialize_non_increasing_p_raises(self):
        # 1. Define test data.
        _x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        _p = np.array([0.0, 0.5, 1.0, 0.5, 1.0])

        # 2. Run test.
        with pytest.raises(ValueError) as exc_info:
            _ = TableDist(_x, _p)

        # 3. Verify expectations.
        assert "Non-exceedance probabilities should be increasing" in str(
            exc_info.value
        )

    def test_compute_cdf(self):
        # 1. Define test data.
        _x = np.array([2.0, 3.0, 5.0])
        _p = np.array([0.0, 0.5, 1.0])
        _dist = TableDist(_x, _p)

        # 2. Run test.
        _cdf_low = _dist.computeCDF(_x[0] - 1.0)
        _cdf_mid = _dist.computeCDF(_x[1])
        _cdf_high = _dist.computeCDF(_x[-1] + 1.0)

        # 3. Verify expectations.
        assert _cdf_low == _p[0]
        assert _cdf_mid == pytest.approx(_p[1], rel=1e-2)
        assert _cdf_high == _p[-1]

    def test_get_mean(self):
        # 1. Define test data.
        _x = np.array([2.0, 3.0, 5.0])
        _p = np.array([0.0, 0.5, 1.0])
        _dist = TableDist(_x, _p)

        # 2. Run test.
        mean = _dist.getMean()

        # 3. Verify expectations.
        assert mean == pytest.approx(_x[1], rel=1e-3)

    def test_get_range(self):
        # 1. Define test data.
        _x = np.array([2.0, 3.0, 5.0])
        _p = np.array([0.0, 0.5, 1.0])
        _dist = TableDist(_x, _p)

        # 2. Run test.
        _range = _dist.getRange()

        # 3. Verify expectations.
        assert _range.getLowerBound()[0] == pytest.approx(_x[0], rel=1e-5)
        assert _range.getUpperBound()[0] == pytest.approx(_x[-1], rel=1e-5)
