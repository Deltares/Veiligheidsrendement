import pytest

from tests import test_data
from tests.test_api import TestApiReportedBugs
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.orm.io.importers.optimization.optimization_measure_result_importer import (
    OptimizationMeasureResultImporter,
)
from vrtool.orm.models.measure_result.measure_result import (
    MeasureResult as OrmMeasureResult,
)
from vrtool.orm.orm_controllers import open_database


class TestOptimizationMeasureResultImporter:

    def test_given_valid_case_import_all_measure_results(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _test_dir_name = "test_stability_multiple_scenarios"
        _test_case_dir = test_data.joinpath(_test_dir_name)
        assert _test_case_dir.exists()

        _investment_years = [0]

        _vrtool_config = TestApiReportedBugs.get_vrtool_config_test_copy(
            _test_case_dir.joinpath("config.json"), request.node.name
        )
        assert not any(_vrtool_config.output_directory.glob("*"))

        # 2. Run test.
        _importer = OptimizationMeasureResultImporter(_vrtool_config, _investment_years)

        with open_database(_vrtool_config.input_database_path).connection_context():
            _imported_results = _importer.import_orm(OrmMeasureResult.select().get())

        # 3. Verify final expectations.
        assert any(_imported_results)
        assert all(isinstance(_ir, MeasureAsInputProtocol) for _ir in _imported_results)
