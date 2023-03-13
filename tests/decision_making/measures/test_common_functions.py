import pandas as pd
import pytest

from tests import test_data
from vrtool.decision_making.measures.common_functions import determine_new_geometry


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )


class TestCommonFunctions:
    params = {
        "test_new_geom": [
            dict(dir="inward", dx=0.5, dy=10, ans1=37.8, ans2=13.06),
            dict(dir="outward", dx=0, dy=20, ans1=91.23, ans2=0.0),
        ],
    }

    def test_new_geom(self, dir, dx, dy, ans1, ans2):
        geometry_change = (dx, dy)
        max_berm_out = 20.0
        test_sheet_new = test_data.joinpath("integrated_SAFE_16-3_small", "DV53.xlsx")
        initial_new = pd.read_excel(
            test_sheet_new,
            sheet_name="Geometry",
            index_col=0,
        )  # New format with BIT BUT etc.
        new_geometry_reinforced = determine_new_geometry(
            geometry_change,
            direction=dir,
            max_berm_out=max_berm_out,
            initial=initial_new,
            berm_height=2,
            geometry_plot=False,
        )
        tol = 0.01
        assert new_geometry_reinforced[1] == pytest.approx(ans1, tol)
        assert new_geometry_reinforced[3] == pytest.approx(ans2, tol)
