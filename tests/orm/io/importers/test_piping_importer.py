import pytest
from peewee import SqliteDatabase

from tests.orm import empty_db_fixture
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.piping_importer import PipingImporter
from vrtool.orm.models import Mechanism
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.mechanism_table import MechanismTable
from vrtool.orm.models.parameter import Parameter
from vrtool.orm.models.section_data import SectionData


class TestPipingImporter:
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

    def _get_mechanism_per_section(self, section: SectionData) -> MechanismPerSection:
        _mechanism = Mechanism.create(name="mechanism")
        return MechanismPerSection.create(section=section, mechanism=_mechanism)

    def _get_valid_computation_scenario(
        self, mechanism_per_section: MechanismPerSection, id: int
    ) -> ComputationScenario:
        _computation_type = ComputationType.create(name=f"irrelevant{id}")
        return ComputationScenario.create(
            mechanism_per_section=mechanism_per_section,
            computation_type=_computation_type,
            computation_name=f"Computation Name {id}",
            scenario_name="Scenario Name",
            scenario_probability=1 - 0.1 * id,
            probability_of_failure=1 - 0.123 * id,
        )

    def _add_computation_scenario_id(
        self, source: list[dict], computation_scenario_id: int
    ) -> None:
        for item in source:
            item["computation_scenario_id"] = computation_scenario_id

    def test_initialize_piping_importer(self):
        _importer = PipingImporter()
        assert isinstance(_importer, PipingImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_piping(self, empty_db_fixture: SqliteDatabase):
        # Setup
        with empty_db_fixture.atomic() as transaction:
            _section_data = self._get_valid_section_data()
            _piping_per_section = self._get_mechanism_per_section(_section_data)
            _computation_scenario1 = self._get_valid_computation_scenario(
                _piping_per_section, 1
            )
            _computation_scenario2 = self._get_valid_computation_scenario(
                _piping_per_section, 2
            )

            parameters1 = [
                {
                    "parameter": "D",
                    "value": 49.0,
                },
                {
                    "parameter": "d70",
                    "value": 0.000226,
                },
                {
                    "parameter": "dh_exit(t)",
                    "value": 0.0051,
                },
            ]
            parameters2 = [
                {
                    "parameter": "D",
                    "value": 41.0,
                },
                {
                    "parameter": "d70",
                    "value": 0.000227,
                },
                {
                    "parameter": "dh_exit(t)",
                    "value": 0.0052,
                },
            ]

            self._add_computation_scenario_id(parameters1, _computation_scenario1.id)
            self._add_computation_scenario_id(parameters2, _computation_scenario2.id)

            Parameter.insert_many(parameters1 + parameters2).execute()
            transaction.commit()

        # 1. Define test data.
        _importer = PipingImporter()

        # 2. Run test
        _mechanism_input = _importer.import_orm(_piping_per_section)

        # 3. Verify expectations.
        assert len(_mechanism_input.input) == 4
        assert _mechanism_input.input["D"][0] == pytest.approx(49.0)
        assert _mechanism_input.input["D"][1] == pytest.approx(41.0)
        assert _mechanism_input.input["d70"][0] == pytest.approx(0.000226)
        assert _mechanism_input.input["d70"][1] == pytest.approx(0.000227)
        assert _mechanism_input.input["P_scenario"][0] == pytest.approx(0.9)
        assert _mechanism_input.input["P_scenario"][1] == pytest.approx(0.8)
        assert _mechanism_input.input["dh_exit(t)"][0] == pytest.approx(0.0051)
        assert _mechanism_input.input["dh_exit(t)"][1] == pytest.approx(0.0052)
        assert len(_mechanism_input.temporals) == 1
        assert _mechanism_input.temporals[0] == "dh_exit(t)"

    def test_import_piping_invalid(self, empty_db_fixture: SqliteDatabase):
        # Setup
        with empty_db_fixture.atomic() as transaction:
            _section_data = self._get_valid_section_data()
            _piping_per_section = self._get_mechanism_per_section(_section_data)
            _computation_scenario1 = self._get_valid_computation_scenario(
                _piping_per_section, 1
            )
            _computation_scenario2 = self._get_valid_computation_scenario(
                _piping_per_section, 2
            )

            parameters1 = [
                # parameter D is only defined for scenario 2, this is not valid
                {
                    "parameter": "d70",
                    "value": 0.000226,
                },
            ]
            parameters2 = [
                {
                    "parameter": "D",
                    "value": 41.0,
                },
                {
                    "parameter": "d70",
                    "value": 0.000227,
                },
            ]

            self._add_computation_scenario_id(parameters1, _computation_scenario1.id)
            self._add_computation_scenario_id(parameters2, _computation_scenario2.id)
            Parameter.insert_many(parameters1 + parameters2).execute()
            transaction.commit()

        # 1. Define test data.
        _importer = PipingImporter()

        # 2. Run test
        with pytest.raises(ValueError) as exception_error:
            _importer.import_orm(_piping_per_section)

        # Assert
        assert str(exception_error.value) == "key not defined for first scenario: D"
