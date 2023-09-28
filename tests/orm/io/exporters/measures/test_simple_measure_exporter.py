from peewee import SqliteDatabase

from tests.orm import empty_db_fixture
from tests.orm.io.exporters.measures import (
    MeasureResultTestInputData,
    MeasureWithDictMocked,
)
from tests.orm.io.exporters.measures.measure_result_test_validators import (
    validate_clean_database,
    validate_measure_result_export,
    validate_no_parameters,
)
from vrtool.orm.io.exporters.measures.simple_measure_exporter import (
    SimpleMeasureExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol


class TestSimpleMeasureExporter:
    def test_initialize(self):
        # Call
        _exporter = SimpleMeasureExporter(None)

        # Assert
        assert isinstance(_exporter, SimpleMeasureExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    def test_export_dom_with_valid_data(self, empty_db_fixture: SqliteDatabase):
        # Setup
        _test_input_data = MeasureResultTestInputData.with_measures_type(
            MeasureWithDictMocked
        )
        validate_clean_database()
        validate_no_parameters(_test_input_data)

        # Call
        _exporter = SimpleMeasureExporter(_test_input_data.measure_per_section)
        _exporter.export_dom(_test_input_data.measure)

        # Assert
        validate_measure_result_export(_test_input_data, {})
