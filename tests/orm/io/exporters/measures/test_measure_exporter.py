from typing import Type
from peewee import SqliteDatabase
import pytest

from tests.orm import empty_db_fixture
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.io.exporters.measures.measure_exporter import MeasureExporter
from vrtool.orm.models.measure_result import MeasureResult
from tests.orm.io.exporters.measures import (
    MeasureResultTestInputData,
    MeasureWithDictMocked,
    MeasureWithListOfDictMocked,
    MeasureWithMeasureResultCollectionMocked,
)
from vrtool.orm.models.measure_result_parameter import MeasureResultParameter


class TestMeasureExporter:
    def test_initialize(self):
        # Call
        _exporter = MeasureExporter(None)

        # Assert
        assert isinstance(_exporter, MeasureExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    @pytest.mark.parametrize(
        "type_measure",
        [
            pytest.param(MeasureWithDictMocked, id="With dictionary"),
            pytest.param(MeasureWithListOfDictMocked, id="With list of dictionaries"),
            pytest.param(
                MeasureWithMeasureResultCollectionMocked,
                id="With Measure Result Collection object",
            ),
        ],
    )
    def test_export_dom_with_valid_data(
        self, type_measure: Type[MeasureProtocol], empty_db_fixture: SqliteDatabase
    ):
        # Setup
        _measures_input_data = MeasureResultTestInputData.with_measures_type(
            type_measure
        )

        assert not any(MeasureResult.select())
        assert not any(MeasureResultParameter.select())

        _exporter = MeasureExporter(_measures_input_data.measure_per_section)

        # Call
        _exporter.export_dom(_measures_input_data.measure)

        # Assert
        _expected_measures = len(_measures_input_data.t_columns)
        assert (
            len(_measures_input_data.measure_per_section.measure_per_section_result)
            == _expected_measures
        )

        assert len(MeasureResult.select()) == _expected_measures
