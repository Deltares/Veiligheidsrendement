from __future__ import annotations
from pathlib import Path
import shutil

import pandas as pd
import pytest
from geolib import DStabilityModel

from tests import test_data, test_externals, test_results
from vrtool.decision_making.measures.common_functions import (
    determine_new_geometry,
    implement_berm_widening,
)

_measure_input = {
    "geometry": pd.DataFrame.from_dict(
        {
            "x": {
                "BUT": -17.0,
                "BUK": 0.0,
                "BIK": 3.5,
                "BBL": 25.0,
                "EBL": 42.0,
                "BIT": 47.0,
                "BIT_0": 47.0,
                "EXT": 3.5,
            },
            "z": {
                "BUT": 4.996,
                "BUK": 10.51,
                "BIK": 10.51,
                "BBL": 6.491,
                "EBL": 5.694,
                "BIT": 5.104,
                "BIT_0": 5.104,
                "EXT": 4.996,
            },
        }
    ),
    "dcrest": 0,
    "dberm": 0,
    "id": 1,
    "StabilityScreen": "yes",
}


class TestCommonFunctions:
    @pytest.mark.parametrize(
        "sheetFileName, direction, dxdy, dcrest_extra, expected_values",
        [
            pytest.param(
                "AW035_087.xlsx",
                "outward",
                (0, 20),
                0,
                (83.67, 0.0),
                id="No Berm case 1a",
            ),
            pytest.param(
                "AW035_087.xlsx",
                "inward",
                (0, 20),
                0,
                (40.00, 20.0),
                id="No Berm case 1b",
            ),
            pytest.param(
                "DV56.xlsx", "outward", (0, 20), 0, (76.82, 0.0), id="No Berm case 2"
            ),
            pytest.param(
                "DV53.xlsx", "inward", (0.5, 10), 0, (37.80, 13.06), id="Inward case"
            ),
            pytest.param(
                "DV53.xlsx", "outward", (0, 20), 0, (91.72, 0.0), id="Outward case"
            ),
            pytest.param(
                "DV86.xlsx",
                "outward",
                (0, 30),
                0,
                (90.375, 10.0),
                id="Outward 30m case",
            ),
            pytest.param(
                "DV53.xlsx",
                "inward",
                (0.5, 10),
                0.4,
                (38.18, 13.37),
                id="Inward case with crest extra",
            ),
        ],
    )
    def test_new_geom(
        self,
        sheetFileName: str,
        direction: str,
        dxdy: tuple[float, float],
        dcrest_extra: float,
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
        if dcrest_extra > 0.0:
            _reinforced_geometry = determine_new_geometry(
                dxdy,
                direction=direction,
                max_berm_out=20.0,
                initial=_traject_test_data,
                berm_height=2,
                geometry_plot=False,
                crest_extra=_traject_test_data["z"].max() - dcrest_extra,
            )
        else:
            _reinforced_geometry = determine_new_geometry(
                dxdy,
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

    def test_implement_berm_widening_dstability_with_screen_generates_intermediate_stix_file(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _berm_input = {
            "STIXNAAM": test_data
            / "stix"
            / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix",
            "DStability_exe_path": test_externals.joinpath("DStabilityConsole"),
        }
        _path_intermediate_stix = test_results / request.node.name
        _expected_file_name = "RW001.+096_STBI_maatgevend_Segment_38005_1D1_ID_1_dberm_0m_dcrest_0m"
        
        if _path_intermediate_stix.exists():
            shutil.rmtree(_path_intermediate_stix)

        # 2. Run test.
        _berm_input_new = implement_berm_widening(
            _berm_input,
            _measure_input,
            measure_parameters={},
            mechanism="StabilityInner",
            computation_type="DStability",
            path_intermediate_stix=_path_intermediate_stix,
            SFincrease=0.2,
            depth_screen=6.0,
        )

        # 3. Verify final expectations.
        _intermediate_stix_file = _berm_input_new["STIXNAAM"]
        assert isinstance(_intermediate_stix_file, Path)
        assert _intermediate_stix_file.stem == _expected_file_name
        assert _intermediate_stix_file.exists()

        # Verify content of the file through the wrapper.
        _dstability_model = DStabilityModel()
        _dstability_model.parse(_intermediate_stix_file)
        assert len(_dstability_model.datastructure.reinforcements) == 2
        assert len(_dstability_model.datastructure.reinforcements[0].ForbiddenLines) == 1
