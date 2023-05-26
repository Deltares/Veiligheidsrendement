import pytest

from peewee import SqliteDatabase

from tests import test_data, test_results
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.solutions_importer import SolutionsImporter
from tests.orm import empty_db_fixture


class TestSolutionsImporter:
    @pytest.fixture
    def valid_config(self) -> VrtoolConfig:
        _vr_config = VrtoolConfig()
        _vr_config.input_directory = test_data
        _vr_config.output_directory = test_results

        yield _vr_config

    def test_initialize(self, valid_config: VrtoolConfig):
        _importer = SolutionsImporter(valid_config, DikeSection())
        assert isinstance(_importer, SolutionsImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_given_no_orm_model_raises_valueerror(self, valid_config: VrtoolConfig):
        # 1. Define test data.
        _importer = SolutionsImporter(valid_config, DikeSection())

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            _importer.import_orm(None)

        # 3. Verify expectations.
        assert str(exc_err.value) == f"No valid value given for SectionData."

    def test_given_section_without_measures_doesnot_raise(self, valid_config: VrtoolConfig, empty_db_fixture: SqliteDatabase):
        pass