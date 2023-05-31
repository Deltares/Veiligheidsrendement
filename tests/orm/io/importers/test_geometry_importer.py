import pandas as pd
import pytest
from peewee import SqliteDatabase

from tests.orm import empty_db_fixture
from vrtool.orm.io.importers.geometry_importer import GeometryImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.characteristic_point_type import CharacteristicPointType
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.profile_point import ProfilePoint
from vrtool.orm.models.section_data import SectionData


class TestGeometryImporter:
    def _get_valid_section_data(self) -> SectionData:
        _test_dike_traject = DikeTrajectInfo.create(traject_name="123")
        return SectionData.create(
            dike_traject=_test_dike_traject,
            section_name="TestSection",
            meas_start=2.4,
            meas_end=4.2,
            section_length=123,
            in_analysis=True,
            crest_height=24,
            annual_crest_decline=42,
        )

    def test_initialize(self):
        _importer = GeometryImporter()
        assert isinstance(_importer, GeometryImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(self, empty_db_fixture: SqliteDatabase):
        # Setup
        point_types = [
            {"name": "BUT"},
            {"name": "BUK"},
            {"name": "BIK"},
            {"name": "BBL"},
            {"name": "EBL"},
            {"name": "BIT"},
        ]

        points = [
            {"x_coordinate": -17, "y_coordinate": 4.996},
            {"x_coordinate": 0, "y_coordinate": 10.939},
            {"x_coordinate": 3.5, "y_coordinate": 10.937},
            {"x_coordinate": 25, "y_coordinate": 6.491},
            {"x_coordinate": 42, "y_coordinate": 5.694},
            {"x_coordinate": 47, "y_coordinate": 5.104},
        ]

        with empty_db_fixture.atomic() as transaction:
            section_data = self._get_valid_section_data()
            CharacteristicPointType.insert_many(point_types).execute()

            for count, point in enumerate(points):
                point["profile_point_type_id"] = CharacteristicPointType.get(
                    CharacteristicPointType.name == point_types[count]["name"]
                ).id
                point["section_data_id"] = section_data.id

            ProfilePoint.insert_many(points).execute()
            transaction.commit()

        _importer = GeometryImporter()

        # Call
        geometry = _importer.import_orm(section_data)

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
        _expected_mssg = "No valid value given for SectionData."

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
