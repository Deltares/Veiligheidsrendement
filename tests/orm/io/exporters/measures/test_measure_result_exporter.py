from typing import Type

from peewee import SqliteDatabase

from tests.orm import empty_db_fixture
from tests.orm.io.exporters.measures import MeasureResultTestInputData
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
from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.measure_result_parameter import MeasureResultParameter


class TestMeasureResultExporter:
    def test_initialize(self):
        _exporter = MeasureResultExporter(None)

        # Verify expectations.
        assert isinstance(_exporter, MeasureResultExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    def test_export_given_valid_mocked_measure_result_returns_expectation(
        self, empty_db_fixture: SqliteDatabase
    ):
        # 1. Define test data.
        class MockedMeasureResult(MeasureResultProtocol):
            def __init__(self) -> None:
                pass

        _test_input_data = MeasureResultTestInputData()
        _test_measure_result = self._initialize_mocked_measure_result(
            MockedMeasureResult, _test_input_data
        )

        assert not any(MeasureResult.select())
        assert not any(MeasureResultParameter.select())

        # 2. Run test.
        MeasureResultExporter(_test_input_data.measure_per_section).export_dom(
            _test_measure_result
        )

        # 3. Verify expectations.
        self._validate_measure_result_export(_test_input_data, {})

    def test_export_given_revetment_measure_section_reliability_creates_measure_parameter_result(
        self, empty_db_fixture: SqliteDatabase
    ):
        # 1. Define test data.
        _test_input_data = MeasureResultTestInputData()
        _test_measure_result = self._initialize_mocked_measure_result(
            RevetmentMeasureSectionReliability, _test_input_data
        )
        _test_measure_result.beta_target = 23.12
        _test_measure_result.transition_level = 2023

        assert not any(MeasureResult.select())
        assert not any(MeasureResultParameter.select())

        # 2. Run test.
        MeasureResultExporter(_test_input_data.measure_per_section).export_dom(
            _test_measure_result
        )

        # 3. Verify expectations.
        self._validate_measure_result_export(
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

    def _validate_measure_result_export(
        self, input_data: MeasureResultTestInputData, parameters_to_validate: dict
    ):
        assert len(MeasureResult.select()) == len(input_data.t_columns)
        assert len(MeasureResultParameter.select()) == len(input_data.t_columns) * len(
            parameters_to_validate
        )
        for year in input_data.t_columns:
            _retrieved_result = MeasureResult.get_or_none(
                (MeasureResult.measure_per_section == input_data.measure_per_section)
                & (MeasureResult.time == year)
            )

            assert isinstance(_retrieved_result, MeasureResult)
            assert (
                _retrieved_result.beta
                == input_data.section_reliability.SectionReliability.loc["Section"][
                    year
                ]
            )
            assert _retrieved_result.cost == input_data.expected_cost
            assert len(_retrieved_result.measure_result_parameters) == len(
                parameters_to_validate.items()
            )

            def measure_result_parameter_exists(name: str, value: float) -> bool:
                return (
                    MeasureResultParameter.select()
                    .where(
                        (MeasureResultParameter.name == name.upper())
                        & (MeasureResultParameter.value == value)
                        & (MeasureResultParameter.measure_result == _retrieved_result)
                    )
                    .exists()
                )

            assert all(
                measure_result_parameter_exists(name, value)
                for name, value in parameters_to_validate.items()
            )
