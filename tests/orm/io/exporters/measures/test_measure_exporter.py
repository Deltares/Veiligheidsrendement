from typing import Type

import pytest
from peewee import SqliteDatabase

from tests.orm import empty_db_fixture, get_basic_measure_per_section
from tests.orm.io.exporters.measures import (
    MeasureResultTestInputData,
    MeasureWithDictMocked,
    MeasureWithListOfDictMocked,
    MeasureWithMeasureResultCollectionMocked,
)
from tests.orm.io.exporters.measures.measure_result_test_validators import (
    validate_clean_database,
    validate_measure_result_export,
    validate_no_parameters,
)
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.orm.io.exporters.measures.measure_exporter import MeasureExporter
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol


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
        # Verify no parameters (except ID) are present as input data.
        validate_no_parameters(_measures_input_data)
        validate_clean_database()

        # Call
        _exporter = MeasureExporter(_measures_input_data.measure_per_section)
        _exporter.export_dom(_measures_input_data.measure)

        # Assert
        validate_measure_result_export(_measures_input_data, {})

    def test_export_dom_invalid_data(self, empty_db_fixture: SqliteDatabase):
        # Setup
        class InvalidMeasureMocked:
            def __init__(self) -> None:
                self.measures = "Cost: 13.37, Reliability: 4.56"

        _measure_per_section = get_basic_measure_per_section()

        validate_clean_database()

        _exporter = MeasureExporter(_measure_per_section)

        # Call
        _measure_to_export = InvalidMeasureMocked()

        with pytest.raises(ValueError) as value_error:
            _exporter.export_dom(_measure_to_export)

        # Assert
        assert str(value_error.value) == "Unknown measure type: InvalidMeasureMocked"

    def test_export_dom_invalid_type(self, empty_db_fixture: SqliteDatabase):
        # Setup
        _measure_per_section = get_basic_measure_per_section()

        validate_clean_database()

        _exporter = MeasureExporter(_measure_per_section)

        # Call
        _measure_to_export = "Cost: 13.37, Reliability: 4.56"

        with pytest.raises(AttributeError) as value_error:
            _exporter.export_dom(_measure_to_export)

        # Assert
        assert str(value_error.value) == "'str' object has no attribute 'measures'"
