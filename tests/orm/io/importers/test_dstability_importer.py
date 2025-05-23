from pathlib import Path
from typing import Callable, Iterator

import pytest
from peewee import SqliteDatabase

from tests.orm.io import add_computation_scenario_id
from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.dstability_importer import DStabilityImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_scenario_parameter import (
    ComputationScenarioParameter,
)
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.supporting_file import SupportingFile


class TestDStabilityImporter:
    @pytest.fixture(name="get_dstability_computation_scenario")
    def _get_valid_computation_scenario(
        self, get_basic_mechanism_per_section: Callable[[], MechanismPerSection]
    ) -> Iterator[Callable[[], ComputationScenario]]:
        def get_dstability_computation_scenario() -> ComputationScenario:
            _mechanism_per_section = get_basic_mechanism_per_section()
            _computation_type = ComputationType.create(
                name=ComputationTypeEnum.DSTABILITY.name
            )
            return ComputationScenario.create(
                mechanism_per_section=_mechanism_per_section,
                computation_type=_computation_type,
                computation_name="Computation Name",
                scenario_name="Scenario Name",
                scenario_probability=1,
                probability_of_failure=1,
            )

        yield get_dstability_computation_scenario

    def test_initialize(self):
        # Setup
        _externals_directory = Path("path/to/externals")
        _stix_directory = Path("path/to/stix")

        # Call
        _importer = DStabilityImporter(_externals_directory, _stix_directory)

        # Assert
        assert isinstance(_importer, DStabilityImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(
        self,
        empty_db_context: SqliteDatabase,
        get_dstability_computation_scenario: Callable[[], ComputationScenario],
    ):
        # Setup
        _supporting_files = [{"filename": "myfile.stix"}]

        parameters = [
            {
                "parameter": "beta",
                "value": 9.13,
            },
            {
                "parameter": "beta2",
                "value": 0.005,
            },
        ]

        with empty_db_context.atomic() as transaction:
            _computation_scenario = get_dstability_computation_scenario()
            add_computation_scenario_id(
                _supporting_files, _computation_scenario.get_id()
            )
            SupportingFile.insert_many(_supporting_files).execute()

            add_computation_scenario_id(parameters, _computation_scenario.get_id())
            ComputationScenarioParameter.insert_many(parameters).execute()

            transaction.commit()

        _externals_directory = Path("path/to/externals")
        _stix_directory = Path("path/to/stix")
        _importer = DStabilityImporter(_externals_directory, _stix_directory)

        # Call
        _mechanism_input = _importer.import_orm(
            _computation_scenario.mechanism_per_section
        )

        # Assert
        assert isinstance(_mechanism_input, MechanismInput)

        assert _mechanism_input.mechanism == MechanismEnum.STABILITY_INNER
        assert len(_mechanism_input.input) == len(parameters) + 2
        assert (
            _mechanism_input.input["STIXNAAM"]
            == _stix_directory / _supporting_files[0]["filename"]
        )
        assert _mechanism_input.input["DStability_exe_path"] == str(
            _externals_directory
        )

        for parameter in parameters:
            assert _mechanism_input.input[parameter.get("parameter")] == pytest.approx(
                parameter.get("value")
            )

    def test_import_orm_with_invalid_computation_type_raises_value_error(
        self,
        empty_db_context: SqliteDatabase,
        get_basic_mechanism_per_section: Callable[[], MechanismPerSection],
    ):
        # Setup
        _supporting_files = [{"filename": "myfile.stix"}]

        with empty_db_context.atomic() as transaction:
            _mechanism_per_section = get_basic_mechanism_per_section()

            _invalid_computation_type = ComputationType.create(
                name=ComputationTypeEnum.INVALID.name
            )
            _computation_scenario = ComputationScenario.create(
                mechanism_per_section=_mechanism_per_section,
                computation_type=_invalid_computation_type,
                computation_name="Computation Name",
                scenario_name="Scenario Name",
                scenario_probability=1,
                probability_of_failure=1,
            )
            add_computation_scenario_id(_supporting_files, _computation_scenario.id)
            SupportingFile.insert_many(_supporting_files).execute()

            transaction.commit()

        _externals_directory = Path("path/to/externals")
        _stix_directory = Path("path/to/stix")
        _importer = DStabilityImporter(_externals_directory, _stix_directory)

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(_computation_scenario.mechanism_per_section)

        # Assert
        _expected_mssg = "All computation types must be 'DSTABILITY'."
        assert str(value_error.value) == _expected_mssg

    def test_import_orm_with_multiple_supporting_files_raises_value_error(
        self,
        empty_db_context: SqliteDatabase,
        get_dstability_computation_scenario: Callable[[], ComputationScenario],
    ):
        # Setup
        _supporting_files = [{"filename": "myfile.stix"}, {"filename": "myfile.stix"}]

        with empty_db_context.atomic() as transaction:
            _computation_scenario = get_dstability_computation_scenario()
            add_computation_scenario_id(_supporting_files, _computation_scenario.id)
            SupportingFile.insert_many(_supporting_files).execute()

            transaction.commit()

        _externals_directory = Path("path/to/externals")
        _stix_directory = Path("path/to/stix")
        _importer = DStabilityImporter(_externals_directory, _stix_directory)

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(_computation_scenario.mechanism_per_section)

        # Assert
        _expected_mssg = "Invalid number of stix files."
        assert str(value_error.value) == _expected_mssg

    def test_import_orm_with_no_supporting_files_raises_value_error(
        self,
        empty_db_context: SqliteDatabase,
        get_dstability_computation_scenario: Callable[[], ComputationScenario],
    ):
        # Setup
        with empty_db_context.atomic() as transaction:
            _computation_scenario = get_dstability_computation_scenario()
            transaction.commit()

        _externals_directory = Path("path/to/externals")
        _stix_directory = Path("path/to/stix")
        _importer = DStabilityImporter(_externals_directory, _stix_directory)

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(_computation_scenario.mechanism_per_section)

        # Assert
        _expected_mssg = "Invalid number of stix files."
        assert str(value_error.value) == _expected_mssg

    def test_import_orm_without_model_raises_value_error(self):
        # Setup
        _externals_directory = Path("path/to/externals")
        _stix_directory = Path("path/to/stix")
        _importer = DStabilityImporter(_externals_directory, _stix_directory)
        _expected_mssg = "No valid value given for MechanismPerSection."

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # Assert
        assert str(value_error.value) == _expected_mssg
