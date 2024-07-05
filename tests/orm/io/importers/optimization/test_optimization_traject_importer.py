from tests import test_data
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.orm.io.importers.optimization.optimization_traject_importer import (
    OptimizationTrajectImporter,
)
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models import DikeTrajectInfo as OrmDikeTrajectInfo
from vrtool.orm.orm_controllers import open_database


class TestOptimizationTrajectImporter:
    def test_initialize(self):
        # 1. Run test.
        _importer = OptimizationTrajectImporter(None, None)

        # 2. Verify expectations
        assert isinstance(_importer, OptimizationTrajectImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_given_traject_with_multiple_sections_imports_all_section_as_input(self):
        """
        [VRTOOL-561] Sections without selected optimization measures.
        """
        # 1. Define test data.
        _test_db = test_data.joinpath(
            "reported_bugs",
            "test_sections_without_selected_measures",
            "database_16-1_testcase_no_custom.db",
        )
        assert _test_db.exists()
        _dummy_config = VrtoolConfig(input_database_name=_test_db.name, traject="16-1")

        # We just need one case to verify the correct functioning
        _measure_results_to_import = [[1, 0]]

        # 2. Run test.
        with open_database(_test_db).connection_context():
            _traject_to_import = OrmDikeTrajectInfo.get(
                OrmDikeTrajectInfo.traject_name == _dummy_config.traject
            )
            _imported_section_as_input = OptimizationTrajectImporter(
                _dummy_config, _measure_results_to_import
            ).import_orm(_traject_to_import)

        # 3. Verify expectations.
        assert isinstance(_imported_section_as_input, list)
        assert len(_imported_section_as_input) == 4
        assert all(
            isinstance(_isai, SectionAsInput) for _isai in _imported_section_as_input
        )
        # Only measures for the second section were imported
        assert not any(_imported_section_as_input[0].measures)
        assert any(_imported_section_as_input[1].measures)
        assert not any(_imported_section_as_input[2].measures)
        assert not any(_imported_section_as_input[3].measures)
