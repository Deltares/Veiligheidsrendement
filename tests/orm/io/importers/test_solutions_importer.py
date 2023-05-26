import pytest
from vrtool.defaults.vrtool_config import VrtoolConfig
from tests import test_data, test_results
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.solutions_importer import SolutionsImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol

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
    
    