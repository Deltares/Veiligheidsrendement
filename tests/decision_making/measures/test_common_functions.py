from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pandas as pd
import pytest
from geolib import DStabilityModel

from tests import test_data, test_externals, test_results
from vrtool.common.enums.mechanism_enum import MechanismEnum
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
        "BUT": [-15, 0.37],
        "BUK": [0, 5.37],
        "BIK": [6, 5.37],
        "BIT": [21, 0.37],
    },
    "DV56": {
        "BIT": [-24, 1.392649],
        "BIK": [-6, 6.41152],
        "BUK": [0, 6.41152],
        "BUT": [14, 1.833213],
    },
    "DV53": {
        "BIT": [-28, 1.253734],
        "EBL": [-22, 3.016411],
        "BBL": [-16, 3.016411],
        "BIK": [-6, 6.616051],
        "BUK": [0, 6.616051],
        "BUT": [18, 1.229376],
    },
    "DV86": {
        "BIT": [-22, 1.287424],
        "BIK": [-14, 4.857041],
        "BUK": [0, 4.857041],
        "BUT": [10, 1.156215],
    },
}


class TestCommonFunctions:
    def _from_dict_to_pd_geometry(self, geom_dict: dict) -> pd.DataFrame:
        def _to_record(geom_item) -> dict:
            return dict(type=geom_item[0], x=geom_item[1][0], z=geom_item[1][1])

        _df_geometry = (
            pd.DataFrame.from_records(map(_to_record, geom_dict.items()))
            .reset_index()
            .set_index("type")
        )
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
            crest_extra=_traject_test_data["z"].max() - dcrest_extra,
        )

        # 3. Verify expectations.
        _tolerance = 0.001
        _reinforcement_1, _reinforcement_3 = expected_values
        assert _reinforced_geometry[1] == pytest.approx(_reinforcement_1, _tolerance)
        assert _reinforced_geometry[3] == pytest.approx(_reinforcement_3, _tolerance)

    def test_geometry_berm_too_high(self):
        # 1. Define test data.
        _geometry_dictionary1 = pd.DataFrame.from_dict(
            {
                "x": {
                    "BUT": -5.14865,
                    "BUK": 0.0,
                    "BIK": 9,
                    "BIT": 33.86232,
                },
                "z": {
                    "BUT": 7.23,
                    "BUK": 8.5,
                    "BIK": 8.5,
                    "BIT": 7.04,
                },
            }
        )

        _geometry_dictionary2 = copy.deepcopy(_geometry_dictionary1)

        # 2. Run test.
        # in this case: berm_height is cut off at 1.45;
        # so berm_heights > 1.45 all give the same answer
        _reinforced_geometry1 = determine_new_geometry(
            (0, 3),
            direction="inward",
            max_berm_out=20.0,
            initial=_geometry_dictionary1,
            berm_height=2.0,
        )

        _reinforced_geometry2 = determine_new_geometry(
            (0, 3),
            direction="inward",
            max_berm_out=20.0,
            initial=_geometry_dictionary2,
            berm_height=3.0,
        )

        assert _reinforced_geometry1[1] == _reinforced_geometry2[1]
        assert _reinforced_geometry1[2] == _reinforced_geometry2[2]
        assert _reinforced_geometry1[3] == _reinforced_geometry2[3]

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
            mechanism=MechanismEnum.STABILITY_INNER,
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
