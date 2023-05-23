import pandas as pd
import pytest
from peewee import SqliteDatabase

from tests.orm.io.importers import db_fixture
from vrtool.orm.io.importers.geometry_importer import GeometryImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.section_data import SectionData


class TestGeometryImporter:
    def test_initialize(self):
        _importer = GeometryImporter()
        assert isinstance(_importer, GeometryImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(self, db_fixture: SqliteDatabase):
        # Setup
        _importer = GeometryImporter()
        section = SectionData.get_by_id(1)

        # Call
        geometry = _importer.import_orm(section.profile_points)

        # Assert
        assert geometry.shape == (6, 2)
        assert list(geometry.columns) == ["x", "z"]

        self._assert_characteristic_point(geometry, "BUT", -17, 4.996)
        self._assert_characteristic_point(geometry, "BUK", 0, 10.939)
        self._assert_characteristic_point(geometry, "BIK", 3.5, 10.937)
        self._assert_characteristic_point(geometry, "BBL", 25, 6.491)
        self._assert_characteristic_point(geometry, "EBL", 42, 5.694)
        self._assert_characteristic_point(geometry, "BIT", 47, 5.104)

    def test_import_orm_without_model_raises_value_error(self):
        # Setup
        _importer = GeometryImporter()
        _expected_mssg = "No valid value given for list."

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # Assert
        assert str(value_error.value) == _expected_mssg

    def _assert_characteristic_point(
        self,
        geometry: pd.DataFrame,
        characteristic_point_name: str,
        expected_x: float,
        expected_z: float,
    ):
        assert geometry.loc[characteristic_point_name]["x"] == pytest.approx(expected_x)
        assert geometry.loc[characteristic_point_name]["z"] == pytest.approx(expected_z)
