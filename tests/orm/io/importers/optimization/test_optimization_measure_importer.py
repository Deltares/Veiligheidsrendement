import pytest
from tests.test_api import TestApiReportedBugs
from vrtool.orm.io.importers.optimization.optimization_measure_importer import (
    OptimizationMeasureImporter,
)
from tests import test_data
from vrtool.orm.models.measure_result.measure_result import (
    MeasureResult as OrmMeasureResult,
)
from vrtool.orm.models.section_data import SectionData as OrmSectionData
from vrtool.orm.orm_controllers import open_database


class TestOptimizationMeasureImporter:

    def test_dummy(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _test_dir_name = "test_stability_multiple_scenarios"
        _test_case_dir = test_data.joinpath(_test_dir_name)
        assert _test_case_dir.exists()

        _vrtool_config = TestApiReportedBugs.get_vrtool_config_test_copy(
            _test_case_dir.joinpath("config.json"), request.node.name
        )
        assert not any(_vrtool_config.output_directory.glob("*"))

        # 2. Run test.
        _importer = OptimizationMeasureImporter(_vrtool_config)

        _test_db = open_database(_vrtool_config.input_database_path)
        _imported_results = _importer.import_orm(OrmMeasureResult.select().get())
        _test_db.close()

        # 3. Verify final expectations.
        assert any(_imported_results)
