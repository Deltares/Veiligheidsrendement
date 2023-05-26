from pathlib import Path
from peewee import SqliteDatabase

from tests import test_data
from tests.orm import empty_db_fixture
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.orm.io.importers.mechanism_reliability_collection_importer import (
    MechanismReliabilityCollectionImporter,
)
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.mechanism_table import MechanismTable
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.supporting_file import SupportingFile


class TestMechanismReliabilityCollectionImporter:
    def _create_valid_config(self):
        _config = VrtoolConfig()
        _config.input_directory = test_data

        return _config

    def _create_valid_scenario(
        self, mechanism_per_section: MechanismPerSection, computation_type: str
    ) -> ComputationScenario:
        _computation_type = ComputationType.create(name=computation_type)
        return ComputationScenario.create(
            mechanism_per_section=mechanism_per_section,
            computation_type=_computation_type,
            computation_name=f"Computation Name",
            scenario_name="Scenario Name",
            scenario_probability=0.9,
            probability_of_failure=0.8,
        )

    def _get_mechanism_per_section_with_scenario(
        self, mechanism: str
    ) -> ComputationScenario:
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

        _mechanism = Mechanism.create(name=mechanism)
        return MechanismPerSection.create(section=_test_section, mechanism=_mechanism)

    def test_import_orm_for_DStability(self, empty_db_fixture: SqliteDatabase):
        # Setup
        _mechanism = "StabilityInner"
        _computation_type = "DSTABILITY"
        _file_name = "something.stix"
        with empty_db_fixture.atomic() as transaction:
            mechanism_per_section = self._get_mechanism_per_section_with_scenario(
                _mechanism
            )
            computation_scenario = self._create_valid_scenario(
                mechanism_per_section, _computation_type
            )

            SupportingFile.create(
                computation_scenario=computation_scenario, filename=_file_name
            )

            transaction.commit()

        _config = self._create_valid_config()
        _config.externals = Path("Path/to/externals")
        _importer = MechanismReliabilityCollectionImporter(_config)

        # Call
        collection = _importer.import_orm(mechanism_per_section)

        # Assert
        self._assert_common_collection_properties(collection, _config)
        self._assert_mechanism_properties(collection, _mechanism, _computation_type)

        mechanism_reliability_input = [
            mechanism_input.Input for mechanism_input in collection.Reliability.values()
        ]
        assert all(
            [
                input.input["DStability_exe_path"] == str(_config.externals)
                for input in mechanism_reliability_input
            ]
        )

    def test_import_orm_for_overflow_hydra_ring(self, empty_db_fixture: SqliteDatabase):
        # Setup
        _mechanism = "Overflow"
        _computation_type = "HRING"
        with empty_db_fixture.atomic() as transaction:
            mechanism_per_section = self._get_mechanism_per_section_with_scenario(
                _mechanism
            )
            computation_scenario = self._create_valid_scenario(
                mechanism_per_section, _computation_type
            )

            MechanismTable.create(
                year=2023,
                value=1.0,
                beta=3.053,
                computation_scenario=computation_scenario,
            )

            transaction.commit()

        _config = self._create_valid_config()
        _importer = MechanismReliabilityCollectionImporter(_config)

        # Call
        collection = _importer.import_orm(mechanism_per_section)

        # Assert
        self._assert_common_collection_properties(collection, _config)
        self._assert_mechanism_properties(collection, _mechanism, _computation_type)

    def test_import_orm_for_piping(self, empty_db_fixture: SqliteDatabase):
        # Setup
        _mechanism = "Piping"
        _computation_type = "SEMIPROB"
        with empty_db_fixture.atomic() as transaction:
            mechanism_per_section = self._get_mechanism_per_section_with_scenario(
                _mechanism
            )
            self._create_valid_scenario(mechanism_per_section, _computation_type)

            transaction.commit()

        _config = self._create_valid_config()
        _importer = MechanismReliabilityCollectionImporter(_config)

        # Call
        collection = _importer.import_orm(mechanism_per_section)

        # Assert
        self._assert_common_collection_properties(collection, _config)
        self._assert_mechanism_properties(collection, _mechanism, _computation_type)

    def test_import_orm_for_stability_inner_simple(
        self, empty_db_fixture: SqliteDatabase
    ):
        # Setup
        _mechanism = "StabilityInner"
        _computation_type = "SIMPLE"
        with empty_db_fixture.atomic() as transaction:
            mechanism_per_section = self._get_mechanism_per_section_with_scenario(
                _mechanism
            )
            self._create_valid_scenario(mechanism_per_section, _computation_type)

            transaction.commit()

        _config = self._create_valid_config()
        _importer = MechanismReliabilityCollectionImporter(_config)

        # Call
        collection = _importer.import_orm(mechanism_per_section)

        # Assert
        self._assert_common_collection_properties(collection, _config)
        self._assert_mechanism_properties(collection, _mechanism, _computation_type)

    def _assert_common_collection_properties(
        self, collection: MechanismReliabilityCollection, config=VrtoolConfig
    ) -> None:
        assert collection.T == config.T
        assert collection.t_0 == config.t_0
        assert list(collection.Reliability.keys()) == [str(year) for year in config.T]

    def _assert_mechanism_properties(
        self,
        collection: MechanismReliabilityCollection,
        mechanism: str,
        computation_type: str,
    ) -> None:
        mechanism_reliabilities = collection.Reliability.values()
        assert all(
            [
                mechanism_reliability.mechanism == mechanism
                and mechanism_reliability.type == computation_type
                for mechanism_reliability in mechanism_reliabilities
            ]
        )
