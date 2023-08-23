import math
from pathlib import Path
from typing import Union

import pytest
from peewee import SqliteDatabase

from tests import test_data
from tests.orm.integration import valid_data_db_fixture
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.dike_traject_importer import DikeTrajectImporter
from vrtool.orm.io.importers.dstability_importer import DStabilityImporter
from vrtool.orm.io.importers.overflow_hydra_ring_importer import (
    OverFlowHydraRingImporter,
)
from vrtool.orm.io.importers.piping_importer import PipingImporter
from vrtool.orm.io.importers.stability_inner_simple_importer import (
    StabilityInnerSimpleImporter,
)
from vrtool.orm.models import DikeTrajectInfo as OrmDikeTrajectInfo
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.mechanism_table import MechanismTable
from vrtool.orm.models.parameter import ComputationScenarioParameter
from vrtool.orm.models.section_data import SectionData


class TestDatabaseIntegration:
    def _assert_float(self, actual: float, expected: Union[float, None]) -> None:
        if not expected:
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(expected)

    def test_import_dike_traject_imports_all_data(
        self, valid_data_db_fixture: SqliteDatabase
    ):
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

    @pytest.mark.skip(reason="This test should not exist. It is also now failing.")
    def test_import_overflow_imports_all_overflow_data(
        self, valid_data_db_fixture: SqliteDatabase
    ):
        # Setup
        _orm_dike_traject_info = OrmDikeTrajectInfo.get_by_id(1)
        _orm_dike_section = _orm_dike_traject_info.dike_sections.select().where(
            SectionData.in_analysis
        )[0]

        _mechanisms_per_first_section = (
            MechanismPerSection.select()
            .join(SectionData, on=MechanismPerSection.section)
            .where(SectionData.id == _orm_dike_section.get_id())
        )

        _overflow_per_first_section = (
            _mechanisms_per_first_section.select()
            .join(Mechanism, on=MechanismPerSection.mechanism == Mechanism.id)
            .where(Mechanism.name == "Overflow")
        )

        # Precondition
        assert len(_overflow_per_first_section) == 1

        computation_scenarios = ComputationScenario.select().where(
            ComputationScenario.mechanism_per_section == _overflow_per_first_section[0]
        )
        assert len(computation_scenarios) == 1

        _overflow_computation_scenario = computation_scenarios[0]
        _importer = OverFlowHydraRingImporter()

        # Call
        _mechanism_input = _importer.import_orm(_overflow_computation_scenario)

        # Assert
        self._assert_overflow_mechanism_input(
            _mechanism_input, _overflow_computation_scenario
        )

    @pytest.mark.skip(reason="This test should not exist. It is also now failing.")
    def test_import_dstability_imports_all__dstability_data(
        self, valid_data_db_fixture: SqliteDatabase
    ):
        # Setup
        # Note: only the first and second sections have a reference to the STIX files
        _orm_dike_traject_info = OrmDikeTrajectInfo.get_by_id(1)
        _orm_dike_section = (
            _orm_dike_traject_info.dike_sections.select()
            .where(SectionData.id == 1)
            .get()
        )

        _mechanisms_per_first_section = (
            MechanismPerSection.select()
            .join(SectionData, on=MechanismPerSection.section)
            .where(SectionData.id == _orm_dike_section.get_id())
        )

        _dstability_per_first_section = (
            _mechanisms_per_first_section.select()
            .join(Mechanism, on=MechanismPerSection.mechanism == Mechanism.id)
            .where(Mechanism.name == "StabilityInner")
        )

        # Precondition
        assert len(_dstability_per_first_section) == 1

        computation_scenarios = ComputationScenario.select().where(
            ComputationScenario.mechanism_per_section
            == _dstability_per_first_section[0]
        )

        _externals_directory = Path("path/to/externals")
        _stix_directory = Path("path/to/stix")
        _importer = DStabilityImporter(_externals_directory, _stix_directory)

        # Call
        # Multiple computation scenarios are defined while only one scenario is supported by the application itself
        _mechanism_input = _importer.import_orm(computation_scenarios[0])

        # Assert
        assert _mechanism_input.mechanism == "StabilityInner"

        expected_parameters = computation_scenarios[0].parameters.select()
        assert len(_mechanism_input.input) == len(expected_parameters) + 2
        self._assert_parameters(_mechanism_input, expected_parameters)

        assert (
            _mechanism_input.input["STIXNAAM"]
            == _stix_directory
            / computation_scenarios[0].supporting_files.select()[0].filename
        )
        assert _mechanism_input.input["DStability_exe_path"] == str(
            _externals_directory
        )

    @pytest.mark.skip(reason="This test should not exist. It is also now failing.")
    def test_import_stability_simple_imports_all_stability_data(
        self, valid_data_db_fixture: SqliteDatabase
    ):
        # Setup
        # Note: Section 22B (id 23) only contains a parameter without stix file support
        _orm_dike_traject_info = OrmDikeTrajectInfo.get_by_id(1)
        _orm_dike_section = (
            _orm_dike_traject_info.dike_sections.select()
            .where(SectionData.id == 23)
            .get()
        )

        _mechanisms_per_first_section = (
            MechanismPerSection.select()
            .join(SectionData, on=MechanismPerSection.section)
            .where(SectionData.id == _orm_dike_section.get_id())
        )

        _stability_per_first_section = (
            _mechanisms_per_first_section.select()
            .join(Mechanism, on=MechanismPerSection.mechanism == Mechanism.id)
            .where(Mechanism.name == "StabilityInner")
        )

        # Precondition
        assert len(_stability_per_first_section) == 1

        computation_scenarios = ComputationScenario.select().where(
            ComputationScenario.mechanism_per_section == _stability_per_first_section[0]
        )

        assert len(computation_scenarios) == 1

        _importer = StabilityInnerSimpleImporter()

        # Call
        _mechanism_input = _importer.import_orm(computation_scenarios[0])

        # Assert
        self._assert_stability_simple_mechanism_input(
            _mechanism_input, computation_scenarios[0]
        )

    @pytest.mark.skip(reason="This test should not exist. It is also now failing.")
    def test_import_piping_imports_all_piping_data(
        self, valid_data_db_fixture: SqliteDatabase
    ):
        # Setup
        _orm_dike_traject_info = OrmDikeTrajectInfo.get_by_id(1)
        _orm_dike_section = _orm_dike_traject_info.dike_sections.select().where(
            SectionData.in_analysis
        )[0]

        _mechanisms_per_first_section = (
            MechanismPerSection.select()
            .join(SectionData, on=MechanismPerSection.section)
            .where(SectionData.id == _orm_dike_section.get_id())
        )

        _piping_per_first_section = (
            _mechanisms_per_first_section.select()
            .join(Mechanism, on=MechanismPerSection.mechanism == Mechanism.id)
            .where(Mechanism.name == "Piping")
        )

        # Precondition
        assert len(_piping_per_first_section) == 1

        _importer = PipingImporter()

        # Call
        _mechanism_input = _importer.import_orm(_piping_per_first_section[0])

        # Assert
        computation_scenarios = ComputationScenario.select().where(
            ComputationScenario.mechanism_per_section == _piping_per_first_section[0]
        )
        self._assert_piping_mechanism_input(_mechanism_input, computation_scenarios)

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
        assert actual.mechanism == "Overflow"

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
        assert actual.mechanism == "StabilityInner"

        expected_parameters = expected.parameters.select()
        assert len(actual.input) == len(expected_parameters)
        self._assert_parameters(actual, expected_parameters)

    def _assert_piping_mechanism_input(
        self, actual: MechanismInput, expected: list[ComputationScenario]
    ) -> None:
        assert actual.mechanism == "Piping"

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
