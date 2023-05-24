import pandas as pd
import pytest
from peewee import SqliteDatabase

from tests import test_data, test_results
from tests.orm.io.importers import db_fixture
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.dike_section_importer import DikeSectionImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.buildings import Buildings
from vrtool.orm.models.section_data import SectionData


class TestDikeSectionImporter:
    @pytest.fixture
    def valid_config(self) -> VrtoolConfig:
        _vr_config = VrtoolConfig()
        _vr_config.input_directory = test_data
        _vr_config.output_directory = test_results

        yield _vr_config

    def test_initialize(self, valid_config: VrtoolConfig):
        _importer = DikeSectionImporter(valid_config)
        assert isinstance(_importer, DikeSectionImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_initialize_without_vrtoolconfig_raises(self):
        with pytest.raises(ValueError) as exc_err:
            DikeSectionImporter(None)

        assert str(exc_err.value) == "VrtoolConfig not provided."

    def test_import_orm(self, db_fixture: SqliteDatabase, valid_config: VrtoolConfig):
        # 1. Define test data.
        _importer = DikeSectionImporter(valid_config)

        # 2. Run test
        _dike_section = _importer.import_orm(SectionData.get_by_id(1))

        # 3. Verify expectations.
        assert isinstance(_dike_section, DikeSection)
        assert _dike_section.name == "section_one"
        assert isinstance(_dike_section.houses, pd.DataFrame)
        assert isinstance(_dike_section.mechanism_data, dict)
        assert any(_dike_section.mechanism_data.items())

        assert isinstance(_dike_section.InitialGeometry, pd.DataFrame)

        section_geometry = _dike_section.InitialGeometry
        assert section_geometry.shape == (6, 2)
        assert list(section_geometry.columns) == ["x", "z"]

        but_point = section_geometry.loc["BUT"]
        assert but_point["x"] == pytest.approx(-17)
        assert but_point["z"] == pytest.approx(4.996)

    def test_import_buildings_list(
        self, db_fixture: SqliteDatabase, valid_config: VrtoolConfig
    ):
        # 1. Define test data.
        _importer = DikeSectionImporter(valid_config)
        _section_data = SectionData.get_by_id(1)
        _buildings_query = Buildings.select().where(
            Buildings.section_data == _section_data
        )

        # 2. Run test
        _buildings_frame = _importer._import_buildings_list(_buildings_query)

        # 3. Verify expectations.
        assert isinstance(_buildings_frame, pd.DataFrame)
        assert list(_buildings_frame.columns) == ["distancefromtoe", "cumulative"]
        assert len(_buildings_frame) == 2
        assert list(_buildings_frame.loc[0]) == [24, 2]
        assert list(_buildings_frame.loc[1]) == [42, 1]

    def test_import_orm_without_model_raises_value(self, valid_config: VrtoolConfig):
        # 1. Define test data.
        _importer = DikeSectionImporter(valid_config)
        _expected_mssg = "No valid value given for SectionData."

        # 2. Run test.
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # 3. Verify expectations.
        assert str(value_error.value) == _expected_mssg
