import filecmp
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely import Polygon

from tests import test_data, test_externals, test_results
from vrtool.decision_making.measures.berm_widening_dstability import (
    BermWideningDStability,
)
from vrtool.failure_mechanisms.stability_inner.dstability_wrapper import (
    DStabilityWrapper,
)

_measure_input_test = {
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
}


class TestBermWideningDStability:
    def test_initialization_berm_widening_dstability_with_valid_input(self):
        # SetUp
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstability_wrapper = DStabilityWrapper(
            _path_test_stix, externals_path=test_externals
        )
        _berm_widening_dstability = BermWideningDStability(
            measure_input=_measure_input_test,
            dstability_wrapper=_dstability_wrapper,
        )

        # Assert instance of BermWideningDStability
        assert isinstance(_berm_widening_dstability, BermWideningDStability)

    def test_berm_widening_dstability_create_new_dstability_model_with_valid_input(
        self,
    ):
        # SetUp
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstability_wrapper = DStabilityWrapper(
            _path_test_stix, externals_path=test_externals
        )
        _berm_widening_dstability = BermWideningDStability(
            measure_input=_measure_input_test,
            dstability_wrapper=_dstability_wrapper,
        )

        path_intermediate_stix = test_results / "test_intermediate_stix"
        if not path_intermediate_stix.exists():
            path_intermediate_stix.mkdir(parents=True)

        path_new_stix = _berm_widening_dstability.create_new_dstability_model(
            path_intermediate_stix
        )

        # Assert
        assert isinstance(path_new_stix, Path)
        assert (
            path_new_stix.parts[-1]
            == "RW001.+096_STBI_maatgevend_Segment_38005_1D1_ID_1_dberm_0m_dcrest_0m.stix"
        )

    @pytest.mark.externals
    @pytest.mark.slow
    def test_when_create_new_dstability_model_then_rerun_with_valid_input(
        self, request: pytest.FixtureRequest
    ):
        # SetUp
        assert test_externals.joinpath(
            "DStabilityConsole"
        ).exists(), "No d-stability console available for testing."

        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstability_wrapper = DStabilityWrapper(
            _path_test_stix, externals_path=test_externals
        )
        _berm_widening_dstability = BermWideningDStability(
            measure_input=_measure_input_test,
            dstability_wrapper=_dstability_wrapper,
        )

        # 1. Run berm widening routine and create the intermediate stix file.
        path_intermediate_stix = test_results / "test_intermediate_stix"
        if not path_intermediate_stix.exists():
            path_intermediate_stix.mkdir(parents=True)

        name_stix = _berm_widening_dstability.create_new_dstability_model(
            path_intermediate_stix
        )

        # 2. Define test data.
        _path_test_stix = path_intermediate_stix / name_stix

        # Create a copy of the file to avoid issues on other tests.
        _test_file = test_results / request.node.name / "file_to_rerun.stix"
        if _test_file.exists():
            shutil.rmtree(_test_file.parent)

        _test_file.parent.mkdir(parents=True)
        shutil.copy(str(_path_test_stix), str(_test_file))

        # 3. Run test.
        _dstability_wrapper.stix_name = _test_file
        _dstability_wrapper.rerun_stix()
        _safety_factor = _dstability_wrapper.get_safety_factor(None)

        # 4. Assert
        assert isinstance(_safety_factor, float)
        assert pytest.approx(1.4588235147100288) == _safety_factor

    def test_find_polygons_to_fill_to_measure_one_polygon_returned(self):
        # SetUp
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstability_wrapper = DStabilityWrapper(
            _path_test_stix, externals_path=test_externals
        )
        _berm_widening_dstability = BermWideningDStability(
            measure_input=_measure_input_test,
            dstability_wrapper=_dstability_wrapper,
        )

        _list_polygons = [Polygon([[0, 0], [2, 0], [2, 1], [0, 1]])]
        _berm_widening_dstability.measure_geometry_points = [[0, 0], [2, 2]]

        # Call
        filling_polygons = _berm_widening_dstability.find_polygons_to_fill_to_measure(
            _list_polygons
        )

        # Assert
        assert isinstance(filling_polygons, list)
        assert len(filling_polygons) == 1
        assert isinstance(filling_polygons[0], Polygon)
        assert filling_polygons[0] == Polygon([[1, 1], [2, 2], [2, 1], [1, 1]])

    def test_find_polygons_to_fill_to_measure_multiple_polygons_returned(self):
        # SetUp
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstability_wrapper = DStabilityWrapper(
            _path_test_stix, externals_path=test_externals
        )
        _berm_widening_dstability = BermWideningDStability(
            measure_input=_measure_input_test,
            dstability_wrapper=_dstability_wrapper,
        )

        _list_polygons = [Polygon([[0, 0], [2, 0], [2, 1], [0, 1]])]
        _berm_widening_dstability.measure_geometry_points = [
            [0, 2],
            [0.5, 1],
            [1.5, 1],
            [2, 2],
        ]

        # Call
        filling_polygons = _berm_widening_dstability.find_polygons_to_fill_to_measure(
            _list_polygons
        )

        # Assert
        assert isinstance(filling_polygons, list)
        assert len(filling_polygons) == 2
        assert isinstance(filling_polygons[0], Polygon)
        assert filling_polygons[0] == Polygon([[0, 1], [0, 2], [0.5, 1], [0, 1]])

    def test_get_modified_meas_geom_with_straight_line(self):
        # SetUp
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstability_wrapper = DStabilityWrapper(
            _path_test_stix, externals_path=test_externals
        )
        _berm_widening_dstability = BermWideningDStability(
            measure_input=_measure_input_test,
            dstability_wrapper=_dstability_wrapper,
        )
        # Call
        measure_geometry_points = _berm_widening_dstability.get_modified_meas_geom(
            straight_line=True
        )

        # Assert
        assert measure_geometry_points == [
            (-17.0, 4.996),
            (0.0, 10.51),
            (3.5, 10.51),
            (25.0, 6.491),
            (42.0, 5.694),
            (47.0, 5.104),
            (100.0, 5.104),
        ]

    def test_get_modified_meas_geom_without_straight_line(self):
        # SetUp
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstability_wrapper = DStabilityWrapper(
            _path_test_stix, externals_path=test_externals
        )
        _berm_widening_dstability = BermWideningDStability(
            measure_input=_measure_input_test,
            dstability_wrapper=_dstability_wrapper,
        )
        # Call
        measure_geometry_points = _berm_widening_dstability.get_modified_meas_geom(
            straight_line=False
        )

        # Assert
        np.testing.assert_almost_equal(
            measure_geometry_points,
            [
                (-26.512148305110998, 1.9107067203304688),
                (-17.0, 4.996),
                (0.0, 10.51),
                (3.5, 10.51),
                (25.0, 6.491),
                (42.0, 5.694),
                (47.0, 5.104),
                (56.931098707062795, 3.932130352566591),
            ],
            decimal=3,
        )

    def test_get_bounding_box(self):
        # SetUp
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )

        _dstability_wrapper = DStabilityWrapper(_path_test_stix, test_externals)
        _measure_input = {"geometry": None, "dcrest": 0, "dberm": 0, "id": 1}
        _berm_widening_dstability = BermWideningDStability(
            measure_input=_measure_input,
            dstability_wrapper=_dstability_wrapper,
        )
        _top_surface = np.array([[-1, 0.5], [2, 0.5]])
        _cropped_polygon_below_surface = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])

        # Call
        bbox = _berm_widening_dstability.get_bounding_box(
            _top_surface, _cropped_polygon_below_surface
        )

        # Assert
        assert isinstance(bbox, Polygon)
        assert bbox.bounds == (0.0, 0.0, 1.0, 0.5)

    def test_adjust_calculation_settings_UpliftVan(self):
        # SetUp
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstability_wrapper = DStabilityWrapper(
            _path_test_stix, externals_path=test_externals
        )
        _berm_widening_dstability = BermWideningDStability(
            measure_input=_measure_input_test,
            dstability_wrapper=_dstability_wrapper,
        )
        _calculation_setting = (
            _dstability_wrapper._dstability_model.datastructure.calculationsettings[-1]
        )
        _berm_widening_dstability.dberm = 2

        # Call
        _berm_widening_dstability.adjust_calculation_settings()

        # Assert
        assert (
            _calculation_setting.UpliftVanParticleSwarm.SearchAreaB.TopLeft.X == 40.026
        )

    def test_adjust_calculation_settings_Bishop(self):
        # SetUp
        _path_test_stix = (
            test_data
            / "stix"
            / "RW001.+096_STBI_maatgevend_Segment_38005_1D1_Bishop.stix"
        )
        _dstability_wrapper = DStabilityWrapper(
            _path_test_stix, externals_path=test_externals
        )
        _berm_widening_dstability = BermWideningDStability(
            measure_input=_measure_input_test,
            dstability_wrapper=_dstability_wrapper,
        )
        _calculation_setting = (
            _dstability_wrapper._dstability_model.datastructure.calculationsettings[-1]
        )
        _berm_widening_dstability.dberm = 2

        # Call
        _berm_widening_dstability.adjust_calculation_settings()

        # Assert
        assert _calculation_setting.BishopBruteForce.SearchGrid.BottomLeft.X == 6.103
        assert (
            _calculation_setting.BishopBruteForce.TangentLines.NumberOfTangentLines
            == 26
        )

    def test_adjust_calculation_settings_invalid_analyse_type_raises_error(self):
        # SetUp
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstability_wrapper = DStabilityWrapper(
            _path_test_stix, externals_path=test_externals
        )
        _berm_widening_dstability = BermWideningDStability(
            measure_input=_measure_input_test,
            dstability_wrapper=_dstability_wrapper,
        )
        _calculation_setting = (
            _dstability_wrapper._dstability_model.datastructure.calculationsettings[-1]
        )
        _calculation_setting.AnalysisType = "Bishop"

        # Call
        with pytest.raises(ValueError) as exception_error:
            _safety_factor = _berm_widening_dstability.adjust_calculation_settings()

        # Assert
        assert str(exception_error.value) == "The analysis type is not supported"
