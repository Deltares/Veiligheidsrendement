from peewee import SqliteDatabase

from tests.orm import empty_db_fixture
from tests.orm.io.exporters.measures import MeasureResultTestInputData
from tests.orm.io.exporters.measures.measure_result_test_validators import (
    validate_clean_database,
    validate_measure_result_export,
)
from vrtool.orm.io.exporters.measures.measure_dict_list_exporter import (
    MeasureDictListExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol


class TestMeasureDictListExporter:
    def test_initialize(self):
        _exporter = MeasureDictListExporter(None)

        # Verify expectations.
        assert isinstance(_exporter, MeasureDictListExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    def test_export_dom_given_valid_composite_measure(
        self, empty_db_fixture: SqliteDatabase
    ):
        # 1. Define test data.
        _input_data = MeasureResultTestInputData()
        _unsupported_param = "unsupported_param"
        _parameters_to_validate = dict(dcrest=4.2, dberm=2.4)
        _measure_with_params = {
            "id": 42,
            "Cost": _input_data.expected_cost,
            "Reliability": _input_data.section_reliability,
            _unsupported_param: 13,
        } | _parameters_to_validate
        validate_clean_database()

        # 2. Run test.
        MeasureDictListExporter(_input_data.measure_per_section).export_dom(
            [_measure_with_params]
        )

        # 3. Verify final expectations.
        validate_measure_result_export(_input_data, _parameters_to_validate)
