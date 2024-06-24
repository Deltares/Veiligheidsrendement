import math
from typing import Iterator, Union

import pytest
from peewee import SqliteDatabase

from tests import test_data
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.dike_traject_importer import DikeTrajectImporter
from vrtool.orm.models import DikeTrajectInfo as OrmDikeTrajectInfo
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_scenario_parameter import (
    ComputationScenarioParameter,
)
from vrtool.orm.models.mechanism_table import MechanismTable
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.orm_controllers import open_database


class TestDatabaseIntegration:
    @pytest.fixture(name="valid_data_db", autouse=False, scope="module")
    def _get_valid_data_db_fixture(self) -> Iterator[SqliteDatabase]:
        _db_file = test_data.joinpath("test_db", "with_valid_data.db")
        assert _db_file.is_file()

        _db = open_database(_db_file)
        assert isinstance(_db, SqliteDatabase)

        yield _db

        _db.close()

    @pytest.mark.usefixtures("valid_data_db")
    def test_import_dike_traject_imports_all_data(self):
        # Setup
        _orm_dike_traject_info = OrmDikeTrajectInfo.get_by_id(1)

        _vr_config = VrtoolConfig()
        _vr_config.input_directory = test_data
        _importer = DikeTrajectImporter(_vr_config)

        # Call
        _dike_traject = _importer.import_orm(_orm_dike_traject_info)

        # Assert
        self._assert_dike_traject_info(
            _dike_traject.general_info, _orm_dike_traject_info
        )

        _orm_dike_sections = _orm_dike_traject_info.dike_sections.select().where(
            SectionData.in_analysis
        )
        assert len(_dike_traject.sections) == len(_orm_dike_sections)

        _first_dike_section = _orm_dike_sections[0]
        self._assert_dike_section(_dike_traject.sections[0], _first_dike_section)

    def _assert_float(self, actual: float, expected: Union[float, None]) -> None:
        if not expected:
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(expected)

    def _assert_dike_traject_info(
        self, actual: DikeTrajectInfo, expected: OrmDikeTrajectInfo
    ) -> None:
        assert actual.traject_name == expected.traject_name

        self._assert_float(actual.omegaPiping, expected.omega_piping)
        self._assert_float(actual.omegaOverflow, expected.omega_overflow)
        self._assert_float(actual.omegaStabilityInner, expected.omega_stability_inner)

        self._assert_float(actual.aPiping, expected.a_piping)
        self._assert_float(actual.bPiping, expected.b_piping)

        self._assert_float(actual.aStabilityInner, expected.a_stability_inner)
        self._assert_float(actual.bStabilityInner, expected.b_stability_inner)

        # Note that currently the value for beta max is a derived value and is therefore not asserted.
        self._assert_float(actual.Pmax, expected.p_max)

        self._assert_float(actual.FloodDamage, expected.flood_damage)
        self._assert_float(actual.TrajectLength, expected.traject_length)

    def _assert_dike_section(self, actual: DikeSection, expected: SectionData) -> None:
        # Note that currently the other columns in the SectionData are not being mapped on the DikeSection.
        assert actual.name == expected.section_name

        for building in expected.buildings_list.select():
            assert actual.houses.loc[building.distance_from_toe][
                "cumulative"
            ] == pytest.approx(building.number_of_buildings)

        _expected_profile_points = expected.profile_points
        assert len(actual.InitialGeometry) == len(_expected_profile_points)

        for expected_profile_point in _expected_profile_points:
            actual_profile_point = actual.InitialGeometry.loc[
                expected_profile_point.profile_point_type.name
            ]
            assert actual_profile_point["x"] == expected_profile_point.x_coordinate
            assert actual_profile_point["z"] == expected_profile_point.y_coordinate

    def _assert_overflow_mechanism_input(
        self, actual: MechanismInput, expected: ComputationScenario
    ) -> None:
        assert actual.mechanism == MechanismEnum.OVERFLOW

        expected_parameters = expected.parameters.select()
        assert len(actual.input) == len(expected_parameters) + 1
        self._assert_parameters(actual, expected_parameters)

        expected_mechanism_table_entries = expected.mechanism_tables.select()
        expected_years = set(
            [str(table_entry.year) for table_entry in expected_mechanism_table_entries]
        )

        actual_crest_height_beta = actual.input["hc_beta"]
        assert len(expected_years.symmetric_difference(actual_crest_height_beta)) == 0
        assert actual_crest_height_beta.index.to_list() == [
            table_entry.value
            for table_entry in expected_mechanism_table_entries.where(
                MechanismTable.year == expected_mechanism_table_entries[0].year
            )
        ]

        for expected_year in iter(expected_years):
            assert list(actual_crest_height_beta[str(expected_year)]) == [
                table_entry.beta
                for table_entry in expected_mechanism_table_entries.where(
                    MechanismTable.year == int(expected_year)
                )
            ]

    def _assert_stability_simple_mechanism_input(
        self, actual: MechanismInput, expected: ComputationScenario
    ) -> None:
        assert actual.mechanism == MechanismEnum.STABILITY_INNER

        expected_parameters = expected.parameters.select()
        assert len(actual.input) == len(expected_parameters)
        self._assert_parameters(actual, expected_parameters)

    def _assert_piping_mechanism_input(
        self, actual: MechanismInput, expected: list[ComputationScenario]
    ) -> None:
        assert actual.mechanism == MechanismEnum.PIPING

        assert all(
            [
                len(input_parameter) == len(expected)
                for input_parameter in actual.input.values()
            ]
        )

        assert len(actual.input) == expected[0].parameters.select().count() + 1

        for count, computation_scenario in enumerate(expected):
            assert actual.input["P_scenario"][count] == pytest.approx(
                computation_scenario.scenario_probability
            )
            for expected_parameter in computation_scenario.parameters.select():
                assert actual.input[expected_parameter.parameter][
                    count
                ] == pytest.approx(expected_parameter.value)

        temporalCnt = (
            expected[0]
            .parameters.select()
            .where(ComputationScenarioParameter.parameter.endswith("(t)"))
            .count()
        )
        assert temporalCnt == len(actual.temporals)

    def _assert_parameters(
        self,
        actual: MechanismInput,
        expected_parameters: list[ComputationScenarioParameter],
    ) -> None:
        for expected_parameter in expected_parameters:
            assert actual.input[expected_parameter.parameter] == pytest.approx(
                expected_parameter.value
            )
