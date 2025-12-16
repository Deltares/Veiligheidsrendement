import numpy as np
import openturns as ot

from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.probabilistic_tools.table_dist import TableDist


class TestLoadInput:

    def test_set_distribution(self):
        # 1. Define test data.
        _load = LoadInput()
        _wls = np.array([0.0, 1.0, 2.0])
        _p = np.array([1.0, 1.5, 2.2])
        _year = 2030
        _gridpoints = 1000

        # 2. Run test.
        _load.set_distribution(_year, _wls, _p, _gridpoints)

        # 3. Verify expectations.
        assert _year in _load.distribution
        assert isinstance(_load.distribution[_year], TableDist)

    def test_compute_h_for_year(self):
        class MockDist(TableDist):
            def computeQuantile(self, p):
                return [3 - p]

        # 1. Define test data.
        _load = LoadInput()
        wls = np.array([0.0, 1.0, 2.0])
        p = np.array([0.0, 0.5, 1.0])
        _load.distribution[2030] = MockDist(wls, p)

        # 2. Run test.
        h = _load.compute_h(2030, 0.2)

        # 3. Verify expectations.
        assert h == 2.8

    def test_compute_h_for_interpolated_year(self):
        class MockDist(ot.PythonDistribution):
            offset: float

            def __init__(self, offset):
                super().__init__(1)
                self.offset = offset

            def computeQuantile(self, p):
                return [3 - p + self.offset]

        # 1. Define test data.
        _load = LoadInput()
        _load.distribution[2030] = MockDist(0)
        _load.distribution[2050] = MockDist(2)

        # 2. Run test.
        h = _load.compute_h(2040, 0.2)

        # 3. Verify expectations.
        assert float(h) == 3.8
