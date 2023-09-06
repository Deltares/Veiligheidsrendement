import pytest
from tests.orm import empty_db_fixture
from peewee import SqliteDatabase
from tests.orm.io.exporters.measures import MeasureResultTestInputData
from tests.orm.io.exporters.measures.test_simple_measure_exporter import (
    MeasureTest,
)
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.exporters.measures.solutions_exporter import SolutionsExporter
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.measure_result_parameter import MeasureResultParameter


class TestSolutionsExporter:
    def test_initialize(self):
        _exporter = SolutionsExporter()

        # Verify expectations.
        assert isinstance(_exporter, SolutionsExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    @pytest.fixture(
        params=[
            pytest.param(MeasureTest(), id="Simple measure"),
        ]
    )
    def solutions_input_data(self, params, empty_db_fixture: SqliteDatabase):
        yield None

    def test_given_solutions_with_supported_measures_raises_error(
        self, solutions_input_data
    ):
        # 1. Define test data.
        _test_input_data = MeasureResultTestInputData()

        _exporter = SolutionsExporter()
        _test_solution = Solutions(DikeSection(), VrtoolConfig())
        _test_solution.measures = []
        assert not any(MeasureResult.select())
        assert not any(MeasureResultParameter.select())

        # 2. Run test.
        _exporter.export_dom(_test_solution)

        # 3. Verify expectations.
        assert len(MeasureResult.select()) == len(_test_input_data)
