from typing import Type
import pytest
from peewee import SqliteDatabase
from tests.orm.io.exporters.measures import (
    MeasureResultTestInputData,
    MeasureWithDictMocked,
    MeasureWithListOfDictMocked,
    MeasureWithMeasureResultCollectionMocked,
)
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
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

    @pytest.mark.parametrize(
        "measure_type",
        [
            pytest.param(MeasureWithDictMocked, id="With dictionary"),
            pytest.param(MeasureWithListOfDictMocked, id="With list of dictionaries"),
            pytest.param(
                MeasureWithMeasureResultCollectionMocked,
                id="With Measure Result Collection object",
            ),
        ],
    )
    def test_given_solutions_with_supported_measures_raises_error(
        self,
        measure_type: Type[MeasureProtocol],
        empty_db_fixture,
    ):
        # 1. Define test data.
        _measures_test_input_data = MeasureResultTestInputData.with_measures_type(
            measure_type
        )
        _exporter = SolutionsExporter()
        _test_solution = Solutions(DikeSection(), VrtoolConfig())
        _test_solution.measures = [_measures_test_input_data.measure]
        assert not any(MeasureResult.select())
        assert not any(MeasureResultParameter.select())

        # 2. Run test.
        _exporter.export_dom(_test_solution)

        # 3. Verify expectations.
        assert any(MeasureResult.select())
        assert len(MeasureResult.select()) == len(_measures_test_input_data.t_columns)
        assert len(
            MeasureResult.select().where(
                MeasureResult.cost == _measures_test_input_data.expected_cost
            )
        ) == len(_measures_test_input_data.t_columns)
