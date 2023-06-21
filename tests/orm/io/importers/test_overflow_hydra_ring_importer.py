import pandas as pd
import pytest
from peewee import SqliteDatabase

from tests.orm import empty_db_fixture
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.overflow_hydra_ring_importer import (
    OverFlowHydraRingImporter,
)
from vrtool.orm.models import Mechanism
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.mechanism_table import MechanismTable
from vrtool.orm.models.section_data import SectionData


class TestOverflowHydraRingImporter:
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
        _importer = OverFlowHydraRingImporter()
        assert isinstance(_importer, OverFlowHydraRingImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(self, empty_db_fixture: SqliteDatabase):
        # Setup
        values = [
            9.51,
            9.76,
            10.01,
        ]

        years = [2023, 2100]
        year_one = years[0]
        mechanism_table_year_one = [
            {
                "year": year_one,
                "value": values[0],
                "beta": 3.053,
            },
            {
                "year": year_one,
                "value": values[1],
                "beta": 3.3826,
            },
            {
                "year": year_one,
                "value": values[2],
                "beta": 3.7217,
            },
        ]

        year_two = years[1]
        mechanism_table_year_two = [
            {
                "year": year_two,
                "value": values[0],
                "beta": 2.2631,
            },
            {
                "year": year_two,
                "value": values[1],
                "beta": 2.5689,
            },
            {
                "year": year_two,
                "value": values[2],
                "beta": 2.9258,
            },
        ]

        with empty_db_fixture.atomic() as transaction:
            _computation_scenario = self._get_valid_computation_scenario()

            _mechanism_tables = mechanism_table_year_one + mechanism_table_year_two
            self._add_computation_scenario_id(
                _mechanism_tables, _computation_scenario.id
            )
            MechanismTable.insert_many(_mechanism_tables).execute()

            transaction.commit()

        _importer = OverFlowHydraRingImporter()

        # Call
        _mechanism_input = _importer.import_orm(_computation_scenario)
        _orm_section_data: SectionData = (
            _computation_scenario.mechanism_per_section.section
        )

        # Assert
        assert isinstance(_mechanism_input, MechanismInput)

        assert _mechanism_input.mechanism == "Overflow"
        assert len(_mechanism_input.input) == 3
        assert _mechanism_input.input["h_crest"] == _orm_section_data.crest_height
        assert (
            _mechanism_input.input["d_crest"] == _orm_section_data.annual_crest_decline
        )

        _mechanism_table_data = _mechanism_input.input["hc_beta"]
        assert _mechanism_input.input["h_crest"] == 24
        assert _mechanism_input.input["d_crest"] == 42
        assert isinstance(_mechanism_table_data, pd.DataFrame)

        assert list(_mechanism_table_data.columns) == [str(year) for year in years]
        assert _mechanism_table_data.index.to_list() == values

        assert list(_mechanism_table_data[str(year_one)]) == [
            entry["beta"] for entry in mechanism_table_year_one
        ]
        assert list(_mechanism_table_data[str(year_two)]) == [
            entry["beta"] for entry in mechanism_table_year_two
        ]

    def test_import_orm_without_model_raises_value_error(self):
        # Setup
        _importer = OverFlowHydraRingImporter()
        _expected_mssg = "No valid value given for ComputationScenario."

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # Assert
        assert str(value_error.value) == _expected_mssg

    def test_import_orm_crest_heights_unequal_raises_value_error(
        self, empty_db_fixture: SqliteDatabase
    ):
        # Setup
        _mechanism_table_source = [
            {
                "year": 2023,
                "value": 1.1,
                "beta": 3.3,
            },
            {
                "year": 2100,
                "value": 2.2,
                "beta": 4.4,
            },
        ]
        with empty_db_fixture.atomic() as transaction:
            _computation_scenario = self._get_valid_computation_scenario()
            self._add_computation_scenario_id(
                _mechanism_table_source, _computation_scenario.id
            )

            MechanismTable.insert_many(_mechanism_table_source).execute()
            transaction.commit()

        _importer = OverFlowHydraRingImporter()

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(_computation_scenario)

        # Assert
        _expected_mssg = f"Crest heights not equal for scenario {_computation_scenario.scenario_name}."
        assert str(value_error.value) == _expected_mssg
