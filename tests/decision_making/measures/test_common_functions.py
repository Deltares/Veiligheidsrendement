from __future__ import annotations
from collections import defaultdict

import shutil
from pathlib import Path

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

_geometry_cases = {
    "AW035_087": {
        "BUT": {"x": -15, "z": 0.37},
        "BUK": {"x": 0, "z": 5.37},
        "BIK": {"x": 6, "z": 5.37},
        "BIT": {"x": 21, "z": 0.37},
    },
    "DV56": {
        "innertoe": {"x": -24, "z": 1.392649},
        "innercrest": {"x": -6, "z": 6.41152},
        "outercrest": {"x": 0, "z": 6.41152},
        "outertoe": {"x": 14, "z": 1.833213},
    },
    "DV53": {
        "BIT": {"x": -28, "z": 1.253734},
        "EBL": {"x": -22, "z": 3.016411},
        "BBL": {"x": -16, "z": 3.016411},
        "BIK": {"x": -6, "z": 6.616051},
        "BUK": {"x": 0, "z": 6.616051},
        "BUT": {"x": 18, "z": 1.229376},
    },
    "DV86": {
        "innertoe": {"x": -22, "z": 1.287424},
        "innercrest": {"x": -14, "z": 4.857041},
        "outercrest": {"x": 0, "z": 4.857041},
        "outertoe": {"x": 10, "z": 1.156215},
    },
}


class TestCommonFunctions:
    def _from_dict_to_pd_geometry(self, geom_dict: dict) -> pd.DataFrame:
        _dict = defaultdict(list)
        for _point_name, _point_values_dict in geom_dict.items():
            _dict["type"].append(_point_name)
            for _coord_name, _coord_value in _point_values_dict.items():
                _dict[_coord_name].append(_coord_value)

        _df_geometry = pd.DataFrame(_dict)
        _df_geometry.set_index("type")
        return _df_geometry

    @pytest.mark.parametrize(
        "geometry_dictionary, direction, dxdy, expected_values",
        [
            pytest.param(
                _geometry_cases["AW035_087"],
                "outward",
                (0, 20),
                (83.67, 0.0),
                id="No Berm - AW035_087",
            ),
            pytest.param(
                _geometry_cases["AW035_087"],
                "inward",
                (0, 20),
                (40.00, 20.0),
                id="No Berm - AW035_087",
            ),
            pytest.param(
                _geometry_cases["DV56"],
                "outward",
                (0, 20),
                (76.82, 0.0),
                id="No Berm - DV56",
            ),
            pytest.param(
                _geometry_cases["DV53"],
                "inward",
                (0.5, 10),
                (37.80, 13.06),
                id="Inward - DV53",
            ),
            pytest.param(
                _geometry_cases["DV53"],
                "outward",
                (0, 20),
                (91.72, 0.0),
                id="Outward - DV53",
            ),
            pytest.param(
                _geometry_cases["DV86"],
                "outward",
                (0, 30),
                (90.375, 10.0),
                id="Outward 30m - DV86",
            ),
        ],
    )
    def test_determine_new_geometry_dcrest_extra_greater_than_zero(
        self,
        geometry_dictionary: dict,
        direction: str,
        dxdy: tuple[float, float],
        expected_values: tuple[float, float],
    ):
        # 1. Define test data.
        _traject_test_data = self._from_dict_to_pd_geometry(geometry_dictionary)

        # 2. Run test.
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

    @pytest.mark.parametrize(
        "geometry_dictionary, direction, dxdy, dcrest_extra, expected_values",
        [
            pytest.param(
                _geometry_cases["DV53"],
                "inward",
                (0.5, 10),
                0.4,
                (38.18, 13.37),
                id="Inward case with crest extra",
            ),
        ],
    )
    def test_determine_new_geometry_dcrest_extra_less_or_equal_than_zero(
        self,
        geometry_dictionary: dict,
        direction: str,
        dxdy: tuple[float, float],
        dcrest_extra: float,
        expected_values: tuple[float, float],
    ):
        # 1. Define test data.
        _traject_test_data = self._from_dict_to_pd_geometry(geometry_dictionary)

        # 2. Run test.
        _reinforced_geometry = determine_new_geometry(
            dxdy,
            direction=direction,
            max_berm_out=20.0,
            initial=_traject_test_data,
            berm_height=2,
            geometry_plot=False,
            crest_extra=_traject_test_data["z"].max() - dcrest_extra,
        )

        # 3. Verify expectations.
        _tolerance = 0.001
        _reinforcement_1, _reinforcement_3 = expected_values
        assert _reinforced_geometry[1] == pytest.approx(_reinforcement_1, _tolerance)
        assert _reinforced_geometry[3] == pytest.approx(_reinforcement_3, _tolerance)

    def test_implement_berm_widening_dstability_with_screen_generates_intermediate_stix_file(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _berm_input = {
            "STIXNAAM": test_data
            / "stix"
            / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix",
            "DStability_exe_path": test_externals.joinpath("DStabilityConsole"),
        }
        _path_intermediate_stix = test_results / request.node.name
        _expected_file_name = (
            "RW001.+096_STBI_maatgevend_Segment_38005_1D1_ID_1_dberm_0m_dcrest_0m"
        )

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
        assert (
            len(_dstability_model.datastructure.reinforcements[0].ForbiddenLines) == 1
        )
