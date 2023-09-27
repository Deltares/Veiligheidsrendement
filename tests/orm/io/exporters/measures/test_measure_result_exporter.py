from typing import Type

from peewee import SqliteDatabase
import pandas as pd
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
from vrtool.orm.models.measure_result import MeasureResult, MeasureResultParameter
from vrtool.orm.models.measure_result.measure_result_mechanism import (
    MeasureResultMechanism,
)
from vrtool.orm.models.measure_result.measure_result_section import MeasureResultSection


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
        # Validate number of entries.
        assert len(MeasureResult.select()) == 1
        _measure_result = MeasureResult.get()
        self._validate_measure_result_parameters(
            _measure_result, parameters_to_validate
        )

        assert len(MeasureResultSection.select()) == len(input_data.t_columns)
        assert len(MeasureResultMechanism.select()) == len(input_data.t_columns) * len(
            input_data.available_mechanisms
        )

        # Validate values.
        for year in input_data.t_columns:
            self._validate_measure_result_section_year(
                _measure_result, input_data, year
            )
            self._validate_measure_result_mechanisms_year(
                _measure_result, input_data, year
            )

    def _validate_measure_result_section_year(
        self,
        measure_result: MeasureResult,
        input_data: MeasureResultTestInputData,
        year: int,
    ):
        _retrieved_result_section = MeasureResultSection.get_or_none(
            (MeasureResultSection.measure_result == measure_result)
            & (MeasureResultSection.time == year)
        )

        assert isinstance(_retrieved_result_section, MeasureResultSection)
        assert (
            _retrieved_result_section.beta
            == input_data.section_reliability.SectionReliability.loc["Section"][year]
        )
        assert _retrieved_result_section.cost == input_data.expected_cost

    def _validate_measure_result_mechanisms_year(
        self,
        measure_result: MeasureResult,
        input_data: MeasureResultTestInputData,
        year: int,
    ):
        for _mechanism_name in input_data.available_mechanisms:
            _retrieved_result_section = MeasureResultMechanism.get_or_none(
                (MeasureResultMechanism.measure_result == measure_result)
                & (MeasureResultMechanism.time == year)
            )

            assert isinstance(_retrieved_result_section, MeasureResultMechanism)
            assert (
                _retrieved_result_section.beta
                == input_data.section_reliability.SectionReliability.loc[
                    _mechanism_name
                ][year]
            )

    def _validate_measure_result_parameters(
        self, measure_result: MeasureResult, parameters_to_validate: dict
    ):
        def measure_result_parameter_exists(name: str, value: float) -> bool:
            return (
                MeasureResultParameter.select()
                .where(
                    (MeasureResultParameter.name == name.upper())
                    & (MeasureResultParameter.value == value)
                    & (MeasureResultParameter.measure_result == measure_result)
                )
                .exists()
            )

        assert len(MeasureResultParameter.select()) == len(parameters_to_validate)
        assert all(
            measure_result_parameter_exists(name, value)
            for name, value in parameters_to_validate.items()
        )
