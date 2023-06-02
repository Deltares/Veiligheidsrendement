from __future__ import annotations

import pytest
from peewee import SqliteDatabase
from pytest import approx

from tests.orm import empty_db_fixture
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.water_level_importer import WaterLevelImporter
from vrtool.orm.models import Mechanism
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.mechanism_table import MechanismTable
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.water_level_data import WaterlevelData

wl1 = [
    {
        "section_data_id": 1,
        "water_level": 2.0,
        "year": 2030,
        "beta": 2.2,
    },
]
wl2 = [
    {
        "section_data_id": 1,
        "water_level": 4.0,
        "year": 2030,
        "beta": 4.4,
    },
]
wl3 = [
    {
        "section_data_id": 1,
        "water_level": 2.3,
        "year": 2050,
        "beta": 2.2,
    },
]
wl4 = [
    {
        "section_data_id": 1,
        "water_level": 4.3,
        "year": 2050,
        "beta": 4.4,
    },
]


class TestWaterLevelImporter:
    def _get_valid_section_data(self) -> SectionData:
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
        return _test_section

    def _get_valid_computation_scenario(
        self, section: SectionData, id: int
    ) -> ComputationScenario:

        _mechanism = Mechanism.create(name="mechanism")
        _mechanism_per_section = MechanismPerSection.create(
            section=section, mechanism=_mechanism
        )

        _computation_type = ComputationType.create(name=f"irrelevant{id}")
        return ComputationScenario.create(
            mechanism_per_section=_mechanism_per_section,
            computation_type=_computation_type,
            computation_name=f"Computation Name {id}",
            scenario_name="Scenario Name",
            scenario_probability=1 - 0.1 * id,
            probability_of_failure=1 - 0.123 * id,
        )

    def get_mechanism_table(self, id: int):
        return [
            {
                "computation_scenario_id": id,
                "year": 2023,
                "value": 1.1,
                "beta": 3.3,
            },
            {
                "computation_scenario_id": id,
                "year": 2100,
                "value": 2.2,
                "beta": 4.4,
            },
        ]

    def test_initialize_water_level_importer(self):
        _importer = WaterLevelImporter(42)
        assert isinstance(_importer, WaterLevelImporter)
        assert isinstance(_importer, OrmImporterProtocol)
        assert _importer.gridpoint == 42

    def test_import_orm_without_no_water_level_data_doesnot_raise(self, empty_db_fixture: SqliteDatabase):
        # 1. Define test data.
        _importer = WaterLevelImporter(42)
        _section_data = self._get_valid_section_data()
        assert not any(_section_data.water_level_data_list)

        # 2. Run test.
        _load_input = _importer.import_orm(_section_data)

        # 3. Verify expectations.
        assert _load_input is None

    @pytest.fixture
    def valid_section_data(
        self, request: pytest.FixtureRequest, empty_db_fixture: SqliteDatabase
    ) -> SectionData:
        with empty_db_fixture.atomic() as transaction:
            _section_data = self._get_valid_section_data()

            _computation_scenario1 = self._get_valid_computation_scenario(
                _section_data, 1
            )

            _mechanism_table_source = self.get_mechanism_table(
                _computation_scenario1.id
            )

            MechanismTable.insert_many(_mechanism_table_source).execute()
            WaterlevelData.insert_many(request.param).execute()
            transaction.commit()
        return _section_data

    @pytest.mark.parametrize(
        "valid_section_data", [pytest.param(wl1 + wl2, id="base case")], indirect=True
    )
    def test_import_water_level(self, valid_section_data: SectionData):
        # 1. Define test data.
        _importer = WaterLevelImporter(1000)

        # 2. Run test
        _load = _importer.import_orm(valid_section_data)

        # 3. Verify expectations.
        sd = _load.distribution[2030].getStandardDeviation()[0]
        assert sd == approx(2.0278)

    @pytest.mark.parametrize(
        "valid_section_data",
        [
            pytest.param(wl2 + wl4 + wl3 + wl1, id="with shuffle"),
            pytest.param(wl1 + wl2 + wl3 + wl4, id="without shuffle"),
        ],
        indirect=True,
    )
    def test_import_water_level_two_years(
        self,
        valid_section_data: SectionData,
    ):

        # 1. Define test data.
        _importer = WaterLevelImporter(1000)

        # 2. Run test
        _load = _importer.import_orm(valid_section_data)

        # 3. Verify expectations.
        sd2030 = _load.distribution[2030].getStandardDeviation()[0]
        assert sd2030 == approx(2.0278)
        sd2050 = _load.distribution[2050].getStandardDeviation()[0]
        assert sd2050 == approx(2.0278)
        mean2030 = _load.distribution[2030].getMean()[0]
        mean2050 = _load.distribution[2050].getMean()[0]
        diff = mean2050 - mean2030
        assert diff == approx(0.3)
