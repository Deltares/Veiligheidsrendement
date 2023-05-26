from peewee import SqliteDatabase
from pytest import approx

from vrtool.orm.io.importers.water_level_importer import WaterLevelImporter
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models import Mechanism
from vrtool.orm.models.mechanism_table import MechanismTable
from vrtool.orm.models.water_level_data import WaterlevelData
from tests.orm import empty_db_fixture


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

    def test_import_water_level(self, empty_db_fixture: SqliteDatabase):
        # Setup
        with empty_db_fixture.atomic() as transaction:
            _section_data = self._get_valid_section_data()

            _computation_scenario1 = self._get_valid_computation_scenario(
                _section_data, 1
            )

            _mechanism_table_source = self.get_mechanism_table(
                _computation_scenario1.id
            )

            wl1 = [
                {
                    "section_data_id": 1,
                    "water_level": 2.0,
                    "year": 2023,
                    "beta": 2.2,
                },
            ]
            wl2 = [
                {
                    "section_data_id": 1,
                    "water_level": 4.0,
                    "year": 2023,
                    "beta": 4.4,
                },
            ]

            MechanismTable.insert_many(_mechanism_table_source).execute()
            WaterlevelData.insert_many(wl1 + wl2).execute()
            transaction.commit()

        # 1. Define test data.
        _importer = WaterLevelImporter()

        # 2. Run test
        _load = _importer.import_orm(_section_data)

        # 3. Verify expectations.
        sd = _load.distribution.getStandardDeviation()[0]
        assert sd == approx(2.0278)
