import pytest
from tests.test_api import TestApiReportedBugs
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.orm.io.importers.optimization.optimization_measure_result_importer import (
    OptimizationMeasureResultImporter,
)
from tests import test_data
from vrtool.orm.models.measure_result.measure_result import (
    MeasureResult as OrmMeasureResult,
)
from vrtool.orm.orm_controllers import (
    import_results_measures_for_optimization,
    open_database,
)


class TestOptimizationMeasureResultImporter:

    def test_dummy(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _test_dir_name = "test_stability_multiple_scenarios"
        _test_case_dir = test_data.joinpath(_test_dir_name)
        assert _test_case_dir.exists()

        _investment_year = 0

        _vrtool_config = TestApiReportedBugs.get_vrtool_config_test_copy(
            _test_case_dir.joinpath("config.json"), request.node.name
        )
        assert not any(_vrtool_config.output_directory.glob("*"))

        # 2. Run test.
        _importer = OptimizationMeasureResultImporter(_vrtool_config, _investment_year)

        _test_db = open_database(_vrtool_config.input_database_path)
        _imported_results = _importer.import_orm(OrmMeasureResult.select().get())
        _test_db.close()

        # 3. Verify final expectations.
        assert any(_imported_results)

    def test_dummy_too(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _test_dir_name = "test_stability_multiple_scenarios"
        _test_case_dir = test_data.joinpath(_test_dir_name)
        assert _test_case_dir.exists()

        _vrtool_config = TestApiReportedBugs.get_vrtool_config_test_copy(
            _test_case_dir.joinpath("config.json"), request.node.name
        )
        assert not any(_vrtool_config.output_directory.glob("*"))

        # 2. Run test.
        with open_database(_vrtool_config.input_database_path).connection_context():
            _imported_data = import_results_measures_for_optimization(
                _vrtool_config, [(omr.id, 0) for omr in OrmMeasureResult.select()]
            )

        # 3. Verify final expectations.
        assert any(_imported_data)
        assert all(
            isinstance(_imp_data, SectionAsInput) for _imp_data in _imported_data
        )
