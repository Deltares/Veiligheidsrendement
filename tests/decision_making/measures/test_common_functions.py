from __future__ import annotations

import pandas as pd
import pytest

from tests import test_data
from vrtool.decision_making.measures.common_functions import determine_new_geometry


class TestCommonFunctions:
    @pytest.mark.parametrize(
        "sheetFileName, direction, dx, dy, expected_values",
        [
            pytest.param(
                "DV56.xlsx", "outward", 0, 20, (91.48, 0.0), id="No Berm case"
            ),
            pytest.param(
                "DV53.xlsx", "inward", 0.5, 10, (37.8, 13.06), id="Inward case"
            ),
            pytest.param(
                "DV53.xlsx", "outward", 0, 20, (91.23, 0.0), id="Outward case"
            ),
        ],
    )
    def test_new_geom(
        self,
        sheetFileName: str,
        direction: str,
        dx: float,
        dy: float,
        expected_values: tuple[float, float],
    ):
        # 1. Define test data.
        _traject_test_file = test_data.joinpath(
            "integrated_SAFE_16-3_small", sheetFileName
        )
        assert _traject_test_file.exists(), "No test file found at {}".format(
            _traject_test_file
        )

        _traject_test_data = pd.read_excel(
            _traject_test_file,
            sheet_name="Geometry",
            index_col=0,
        )

        # 2. Run test.
        _reinforced_geometry = determine_new_geometry(
            (dx, dy),
            direction=direction,
            max_berm_out=20.0,
            initial=_traject_test_data,
            berm_height=2,
            geometry_plot=False,
        )

        # 3. Verify expectations.
        _tolerance = 0.001
        _reinforcement_1, _reinforcement_3 = expected_values
        assert _reinforced_geometry[1] == pytest.approx(_reinforcement_1, _tolerance)
        assert _reinforced_geometry[3] == pytest.approx(_reinforcement_3, _tolerance)
