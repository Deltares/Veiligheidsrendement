from typing import Type

import pytest
from pandas import DataFrame
from peewee import SqliteDatabase

from tests.orm import empty_db_fixture, get_basic_measure_per_section
from tests.orm.io.exporters.measures.measure_result_test_validators import (
    MeasureResultTestInputData,
    MeasureWithDictMocked,
    MeasureWithListOfDictMocked,
    MeasureWithMeasureResultCollectionMocked,
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
        "unsupported_parameters",
        [
            pytest.param(dict(), id="WITHOUT unsupported parameters present"),
            pytest.param(
                dict(
                    geometry=DataFrame.from_dict(
                        {"x_coord": [0.24, 0.42], "y_coord": [2.4, 4.2]}
                    )
                ),
                id="WITH unsupported parameters present",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "parameters_to_validate",
        [
            pytest.param(dict(), id="Without supported parameters"),
            pytest.param(dict(dcrest=4.2, dberm=2.4), id="With supported parameters"),
        ],
    )
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
        self,
        type_measure: Type[MeasureProtocol],
        parameters_to_validate: dict,
        unsupported_parameters: dict,
        empty_db_fixture: SqliteDatabase,
    ):
        # Setup
        _measures_input_data = MeasureResultTestInputData.with_measures_type(
            type_measure, parameters_to_validate | unsupported_parameters
        )
        # Verify no parameters (except ID) are present as input data.
        if not parameters_to_validate:
            validate_no_parameters(_measures_input_data)
        validate_clean_database()

        # Call
        _exporter = MeasureExporter(_measures_input_data.measure_per_section)
        _exporter.export_dom(_measures_input_data.measure)

        # Assert
        validate_measure_result_export(
            _measures_input_data, _measures_input_data.parameters_to_validate
        )

    def test_export_dom_given_valid_measure_dict_list(
        self, empty_db_fixture: SqliteDatabase
    ):
        # 1. Define test data.
        _unsupported_param = "unsupported_param"
        _parameters_to_validate = dict(dcrest=4.2, dberm=2.4)
        _input_data = MeasureResultTestInputData.with_measures_type(
            MeasureWithListOfDictMocked, _parameters_to_validate
        )

        validate_clean_database()

        # 2. Run test.
        MeasureExporter(_input_data.measure_per_section).export_dom(_input_data.measure)

        # 3. Verify final expectations.
        validate_measure_result_export(_input_data, _input_data.parameters_to_validate)

    def test_export_dom_given_dict_measure(self, empty_db_fixture: SqliteDatabase):
        # Setup
        _test_input_data = MeasureResultTestInputData.with_measures_type(
            MeasureWithDictMocked, {}
        )
        validate_clean_database()
        validate_no_parameters(_test_input_data)

        # Call
        _exporter = MeasureExporter(_test_input_data.measure_per_section)
        _exporter.export_dom(_test_input_data.measure)

        # Assert
        validate_measure_result_export(
            _test_input_data, _test_input_data.parameters_to_validate
        )

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

        with pytest.raises(TypeError) as value_error:
            _exporter.export_dom(_measure_to_export)

        # Assert
        assert str(value_error.value) == "Unknown measure type: 'InvalidMeasureMocked'."

    def test_export_dom_invalid_type(self, empty_db_fixture: SqliteDatabase):
        # Setup
        _measure_per_section = get_basic_measure_per_section()

        validate_clean_database()

        _exporter = MeasureExporter(_measure_per_section)

        # Call
        _measure_to_export = "Cost: 13.37, Reliability: 4.56"

        with pytest.raises(TypeError) as value_error:
            _exporter.export_dom(_measure_to_export)

        # Assert
        assert str(value_error.value) == "Unknown measure type: 'str'."
