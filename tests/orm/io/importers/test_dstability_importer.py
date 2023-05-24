import pytest

from peewee import SqliteDatabase

from tests.orm import empty_db_fixture
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.dstability_importer import DStabilityImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.supporting_file import SupportingFile


class TestDStabilityImporter:
    def _get_valid_computation_scenario(self) -> ComputationScenario:
        _test_dike_traject = DikeTrajectInfo.create(traject_name="123")
        _test_section = SectionData.create(
            dike_traject=_test_dike_traject,
            section_name="TestSection",
            meas_start=2.4,
            meas_end=4.2,
            section_length=123,
            in_analysis=True,
            crest_height=24,
            annual_crest_decline=42,
        )

        _mechanism = Mechanism.create(name="mechanism")
        _mechanism_per_section = MechanismPerSection.create(
            section=_test_section, mechanism=_mechanism
        )

        _computation_type = ComputationType.create(name="irrelevant")
        return ComputationScenario.create(
            mechanism_per_section=_mechanism_per_section,
            computation_type=_computation_type,
            computation_name="Computation Name",
            scenario_name="Scenario Name",
            scenario_probability=1,
            probability_of_failure=1,
        )

    def _add_computation_scenario_id(
        self, source: list[dict], computation_scenario_id: int
    ) -> None:
        for item in source:
            item["computation_scenario_id"] = computation_scenario_id

    def test_initialize(self):
        _importer = DStabilityImporter()
        assert isinstance(_importer, DStabilityImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(self, empty_db_fixture: SqliteDatabase):
        # Setup
        _supporting_files = [{"filename": "myfile.stix"}]

        with empty_db_fixture.atomic() as transaction:
            _computation_scenario = self._get_valid_computation_scenario()
            self._add_computation_scenario_id(
                _supporting_files, _computation_scenario.id
            )
            SupportingFile.insert_many(_supporting_files).execute()

            transaction.commit()

        _importer = DStabilityImporter()

        # Call
        _mechanism_input = _importer.import_orm(_computation_scenario)

        # Assert
        assert isinstance(_mechanism_input, MechanismInput)

        assert _mechanism_input.mechanism == "StabilityInner"
        assert len(_mechanism_input.input) == 1
        assert _mechanism_input.input["stix_file"] == _supporting_files[0]["filename"]

    def test_import_orm_with_multiple_supporting_files_raises_value_error(
        self, empty_db_fixture: SqliteDatabase
    ):
        # Setup
        _supporting_files = [{"filename": "myfile.stix"}, {"filename": "myfile.stix"}]

        with empty_db_fixture.atomic() as transaction:
            _computation_scenario = self._get_valid_computation_scenario()
            self._add_computation_scenario_id(
                _supporting_files, _computation_scenario.id
            )
            SupportingFile.insert_many(_supporting_files).execute()

            transaction.commit()

        _importer = DStabilityImporter()

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(_computation_scenario)

        # Assert
        _expected_mssg = "Invalid number of stix files."
        assert str(value_error.value) == _expected_mssg

    def test_import_orm_without_model_raises_value_error(self):
        # Setup
        _importer = DStabilityImporter()
        _expected_mssg = "No valid value given for ComputationScenario."

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # Assert
        assert str(value_error.value) == _expected_mssg
