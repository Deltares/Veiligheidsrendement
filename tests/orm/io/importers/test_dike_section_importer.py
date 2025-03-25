from typing import Iterator

import pandas as pd
import pytest
from peewee import SqliteDatabase

from tests import test_data, test_results
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.dike_section_importer import DikeSectionImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.buildings import Buildings
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.orm_controllers import open_database


class TestDikeSectionImporter:
    @pytest.fixture(name="valid_config")
    def _get_valid_config_fixture(self) -> Iterator[VrtoolConfig]:
        _vr_config = VrtoolConfig()
        _vr_config.input_directory = test_data
        _vr_config.output_directory = test_results

        yield _vr_config

    @pytest.fixture(name="db_fixture", autouse=False, scope="module")
    def _get_db_fixture(self) -> Iterator[SqliteDatabase]:
        _db_file = test_data.joinpath("test_db", "vrtool_db.db")
        assert _db_file.is_file()

        _db = open_database(_db_file)
        assert isinstance(_db, SqliteDatabase)

        yield _db

        _db.close()

    def test_initialize(self, valid_config: VrtoolConfig):
        _importer = DikeSectionImporter(valid_config)
        assert isinstance(_importer, DikeSectionImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_initialize_without_vrtoolconfig_raises(self):
        with pytest.raises(ValueError) as exc_err:
            DikeSectionImporter(None)

        assert str(exc_err.value) == "VrtoolConfig not provided."

    @pytest.mark.usefixtures("db_fixture")
    def test_import_orm(self, valid_config: VrtoolConfig):
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

    @pytest.mark.usefixtures("db_fixture")
    def test__import_buildings_list(self, valid_config: VrtoolConfig):
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
        assert list(_buildings_frame.columns) == ["cumulative"]
        assert _buildings_frame.index.name == "distancefromtoe"
        assert len(_buildings_frame) == 2
        assert list(_buildings_frame.loc[24]) == [2]
        assert list(_buildings_frame.loc[42]) == [1]
    
    @pytest.fixture(name="valid_section_data", scope="module")
    def _get_valid_section_data_fixture(self) -> Iterator[SectionData]:
        _db_file = test_data.joinpath("test_db", "vrtool_with_filtered_results.db")
        assert _db_file.is_file()

        _db = open_database(_db_file)
        assert isinstance(_db, SqliteDatabase)

        yield SectionData.get_by_id(1)

        _db.close()


    @pytest.mark.parametrize("excluded_mechanism", [
        MechanismEnum.REVETMENT,
        MechanismEnum.PIPING,
        MechanismEnum.STABILITY_INNER,
        ])
    def test__get_mechanism_reliability_collection_list(
        self,
        excluded_mechanism: MechanismEnum,
        valid_config: VrtoolConfig,
        valid_section_data: SectionData,
    ):
        # 1. Define test data.
        valid_config.excluded_mechanisms.append(excluded_mechanism)
        _importer = DikeSectionImporter(valid_config)

        # 2. Run test.
        _result = _importer._get_mechanism_reliability_collection_list(valid_section_data)

        # 3. Verify expectations.
        assert isinstance(_result, list)
        assert not any([_mrc.mechanism == excluded_mechanism for _mrc in _result])
        assert all([_mrc.mechanism in valid_config.supported_mechanisms for _mrc in _result])


    def test_import_orm_without_model_raises_value(self, valid_config: VrtoolConfig):
        # 1. Define test data.
        _importer = DikeSectionImporter(valid_config)
        _expected_mssg = "No valid value given for SectionData."

        # 2. Run test.
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # 3. Verify expectations.
        assert str(value_error.value) == _expected_mssg
