from __future__ import annotations

from typing import Callable

import pytest
from peewee import SqliteDatabase

from tests import test_data, test_externals
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


class TestDataHelper:
    @staticmethod
    def create_valid_config():
        _config = VrtoolConfig()
        _config.input_directory = test_data
        _config.externals = test_externals

        return _config

    @staticmethod
    def _create_valid_scenario(
        mechanism_per_section: MechanismPerSection, computation_type: str
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

    @staticmethod
    def get_mechanism_per_section_with_scenario(mechanism: str) -> ComputationScenario:
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

    @staticmethod
    def get_valid_mechanism_per_section(
        mechanism: str, computation_type: str
    ) -> MechanismPerSection:
        mechanism_per_section = TestDataHelper.get_mechanism_per_section_with_scenario(
            mechanism
        )
        TestDataHelper._create_valid_scenario(mechanism_per_section, computation_type)

        return mechanism_per_section

    @staticmethod
    def get_mechanism_per_section_with_supporting_file(
        mechanism: str, computation_type: str
    ) -> MechanismPerSection:
        _file_name = "something.stix"
        mechanism_per_section = TestDataHelper.get_mechanism_per_section_with_scenario(
            mechanism
        )
        computation_scenario = TestDataHelper._create_valid_scenario(
            mechanism_per_section, computation_type
        )

        SupportingFile.create(
            computation_scenario=computation_scenario, filename=_file_name
        )
        return mechanism_per_section

    @staticmethod
    def get_overflow_hydraring_mechanism_per_section(
        mechanism: str, computation_type: str
    ) -> MechanismPerSection:
        mechanism_per_section = TestDataHelper.get_mechanism_per_section_with_scenario(
            mechanism
        )
        computation_scenario = TestDataHelper._create_valid_scenario(
            mechanism_per_section, computation_type
        )

        MechanismTable.create(
            year=2023,
            value=1.0,
            beta=3.053,
            computation_scenario=computation_scenario,
        )
        return mechanism_per_section


class TestMechanismReliabilityCollectionImporter:
    def test_import_orm_for_dstability(self, empty_db_fixture: SqliteDatabase):
        # Setup
        _mechanism = "StabilityInner"
        _computation_type = "DSTABILITY"
        _config = TestDataHelper.create_valid_config()
        _importer = MechanismReliabilityCollectionImporter(_config)
        _mechanism_per_section = (
            TestDataHelper.get_mechanism_per_section_with_supporting_file(
                _mechanism, _computation_type
            )
        )

        # Call
        collection = _importer.import_orm(_mechanism_per_section)

        # Assert
        self._assert_common_collection_properties(collection, _config)
        self._assert_mechanism_properties(collection, _mechanism, _computation_type)

        mechanism_reliability_input = [
            mechanism_input.Input for mechanism_input in collection.Reliability.values()
        ]
        assert all(
            [
                _mr_input.input["DStability_exe_path"] == str(_config.externals)
                for _mr_input in mechanism_reliability_input
            ]
        )

    @pytest.mark.parametrize(
        "mechanism, computation_type, get_mechanism_per_section",
        [
            pytest.param(
                "StabilityInner",
                "SIMPLE",
                TestDataHelper.get_valid_mechanism_per_section,
                id="Stability Inner simple",
            ),
            pytest.param(
                "Piping",
                "SEMIPROB",
                TestDataHelper.get_valid_mechanism_per_section,
                id="Piping SEMIPROB",
            ),
            pytest.param(
                "Overflow",
                "HRING",
                TestDataHelper.get_overflow_hydraring_mechanism_per_section,
                id="Overflow HRING",
            ),
        ],
    )
    def test_import_orm_with_simple_mechanism_per_section(
        self,
        mechanism: str,
        computation_type: str,
        get_mechanism_per_section: Callable,
        empty_db_fixture: SqliteDatabase,
    ):
        # Setup
        _config = TestDataHelper.create_valid_config()
        _importer = MechanismReliabilityCollectionImporter(_config)
        _mechanism_per_section = get_mechanism_per_section(mechanism, computation_type)

        # Call
        collection = _importer.import_orm(_mechanism_per_section)

        # Assert
        self._assert_common_collection_properties(collection, _config)
        self._assert_mechanism_properties(collection, mechanism, computation_type)

    def test_import_orm_without_model_raises_value_error(self):
        # Setup
        _config = TestDataHelper.create_valid_config()
        _importer = MechanismReliabilityCollectionImporter(_config)
        _expected_mssg = "No valid value given for MechanismPerSection."

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # Assert
        assert str(value_error.value) == _expected_mssg

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
