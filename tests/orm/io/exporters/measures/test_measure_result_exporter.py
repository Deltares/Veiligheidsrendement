from typing import Type

from tests.orm import with_empty_db_context
from tests.orm.io.exporters.measures.measure_result_test_validators import (
    MeasureResultTestInputData,
    validate_clean_database,
    validate_measure_result_export,
)
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultProtocol,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_section_reliability import (
    RevetmentMeasureSectionReliability,
)
from vrtool.orm.io.exporters.measures.measure_result_exporter import (
    MeasureResultExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol


class TestMeasureResultExporter:
    def test_initialize(self):
        _exporter = MeasureResultExporter(None)

        # Verify expectations.
        assert isinstance(_exporter, MeasureResultExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    @with_empty_db_context
    def test_export_given_valid_mocked_measure_result_returns_expectation(self):
        # 1. Define test data.
        class MockedMeasureResult(MeasureResultProtocol):
            def __init__(self) -> None:
                pass

            def get_measure_result_parameters(self) -> dict:
                return {}

        _test_input_data = MeasureResultTestInputData()
        _test_measure_result = self._initialize_mocked_measure_result(
            MockedMeasureResult, _test_input_data
        )
        validate_clean_database()

        # 2. Run test.
        MeasureResultExporter(_test_input_data.measure_per_section).export_dom(
            _test_measure_result
        )

        # 3. Verify expectations.
        validate_measure_result_export(_test_input_data, {})

    @with_empty_db_context
    def test_export_given_revetment_measure_section_reliability_creates_measure_parameter_result(
        self,
    ):
        # 1. Define test data.
        _test_input_data = MeasureResultTestInputData()
        _test_measure_result = self._initialize_mocked_measure_result(
            RevetmentMeasureSectionReliability, _test_input_data
        )
        _test_measure_result.beta_target = 23.12
        _test_measure_result.transition_level = 2023

        validate_clean_database()

        # 2. Run test.
        MeasureResultExporter(_test_input_data.measure_per_section).export_dom(
            _test_measure_result
        )

        # 3. Verify expectations.
        validate_measure_result_export(
            _test_input_data, dict(beta_target=23.12, transition_level=2023)
        )

    def _initialize_mocked_measure_result(
        self,
        measure_type: Type[MeasureResultProtocol],
        input_data: MeasureResultTestInputData,
    ):
        _test_measure_result = measure_type()
        _test_measure_result.measure_id = "A mocked measure"
        _test_measure_result.section_reliability = input_data.section_reliability
        _test_measure_result.cost = input_data.expected_cost
        return _test_measure_result
