import pandas as pd
import pytest

from vrtool.decision_making.measures.common_functions import determine_new_geometry
from tests import test_data


class TestCommonFunctions:
    def test_new_geom_inward(self):
        # Inward comparison (can be done for several geometry changes). And we can also include initial geometries without a berm.
        geometry_change = (0.5, 10.0)
        max_berm_out = 20.0
        test_sheet_new = test_data.joinpath(
            "integrated_SAFE_16-3_small", "DV53.xlsx"
        )
        initial_new = pd.read_excel(
            test_sheet_new,
            sheet_name="Geometry",
            index_col=0,
        )  # New format with BIT BUT etc.
        new_geometry_reinforced = determine_new_geometry(
            geometry_change,
            direction="inward",
            max_berm_out=max_berm_out,
            initial=initial_new,
            berm_height=2,
            geometry_plot=False,
        )
        assert new_geometry_reinforced[1] == pytest.approx(37.8, 0.01)
        assert new_geometry_reinforced[3] == pytest.approx(13.06, 0.01)

    def test_new_geom_berm1(self):

        # Outward comparison berm 1
        # NB: result is incorrect, should be (almost) same as outward berm2
        geometry_change = (0.0, 20.0)
        max_berm_out = 20.0

        test_sheet_new = test_data.joinpath(
            "integrated_SAFE_16-3_small", "DV53_new.xlsx"
        )
        initial_new = pd.read_excel(
            test_sheet_new,
            sheet_name="Geometry",
            index_col=0,
        )  # New format with BIT BUT etc.
        new_geometry_reinforced = determine_new_geometry(
            geometry_change,
            direction="outward",
            max_berm_out=max_berm_out,
            initial=initial_new,
            berm_height=2,
            geometry_plot=False,
        )

        assert new_geometry_reinforced[1] == pytest.approx(91.23, 0.01)
        assert new_geometry_reinforced[3] == pytest.approx(0.0, 0.01)
