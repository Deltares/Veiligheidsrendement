from collections.abc import Callable

import numpy as np
import pytest
from peewee import SqliteDatabase

from tests.orm.io import add_computation_scenario_id
from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.piping_importer import PipingImporter
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_scenario_parameter import (
    ComputationScenarioParameter,
)
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.mechanism_per_section import MechanismPerSection


class TestPipingImporter:
    def _get_valid_computation_scenario(
        self, mechanism_per_section: MechanismPerSection, id: int
    ) -> ComputationScenario:
        _computation_type = ComputationType.create(name=ComputationTypeEnum(id).name)
        return ComputationScenario.create(
            mechanism_per_section=mechanism_per_section,
            computation_type=_computation_type,
            computation_name=f"Computation Name {id}",
            scenario_name="Scenario Name",
            scenario_probability=1 - 0.1 * id,
            probability_of_failure=1 - 0.123 * id,
        )

    def test_initialize_piping_importer(self):
        _importer = PipingImporter()
        assert isinstance(_importer, PipingImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_piping(
        self,
        empty_db_context: SqliteDatabase,
        get_basic_mechanism_per_section: Callable[[], MechanismPerSection],
    ):
        # Setup
        with empty_db_context.atomic() as transaction:
            _piping_per_section = get_basic_mechanism_per_section()
            _computation_scenario1 = self._get_valid_computation_scenario(
                _piping_per_section, 1
            )
            _computation_scenario2 = self._get_valid_computation_scenario(
                _piping_per_section, 2
            )

            parameters1 = [
                {
                    "parameter": "d_wvp",
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
                    "parameter": "d_wvp",
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

            add_computation_scenario_id(parameters1, _computation_scenario1.id)
            add_computation_scenario_id(parameters2, _computation_scenario2.id)

            ComputationScenarioParameter.insert_many(
                parameters1 + parameters2
            ).execute()
            transaction.commit()

        # 1. Define test data.
        _importer = PipingImporter()

        # 2. Run test
        _mechanism_input = _importer.import_orm(_piping_per_section)

        # 3. Verify expectations.
        assert len(_mechanism_input.input) == 7
        assert _mechanism_input.input["Scenario"] == [
            _computation_scenario1.scenario_name,
            _computation_scenario2.scenario_name,
        ]
        assert _mechanism_input.input["d_wvp"][0] == pytest.approx(49.0)
        assert _mechanism_input.input["d_wvp"][1] == pytest.approx(41.0)
        assert _mechanism_input.input["d70"][0] == pytest.approx(0.000226)
        assert _mechanism_input.input["d70"][1] == pytest.approx(0.000227)
        assert _mechanism_input.input["P_scenario"][0] == pytest.approx(0.9)
        assert _mechanism_input.input["P_scenario"][1] == pytest.approx(0.8)
        assert _mechanism_input.input["dh_exit(t)"][0] == pytest.approx(0.0051)
        assert _mechanism_input.input["dh_exit(t)"][1] == pytest.approx(0.0052)
        assert _mechanism_input.input["Pf"] == 0.754
        assert np.array_equal(_mechanism_input.input["Beta"], np.array([0, 0]))
        assert len(_mechanism_input.temporals) == 1
        assert _mechanism_input.temporals[0] == "dh_exit(t)"

    def test_import_piping_invalid(
        self,
        empty_db_context: SqliteDatabase,
        get_basic_mechanism_per_section: Callable[[], MechanismPerSection],
    ):
        # Setup
        with empty_db_context.atomic() as transaction:
            _piping_per_section = get_basic_mechanism_per_section()
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
                    "parameter": "d_wvp",
                    "value": 41.0,
                },
                {
                    "parameter": "d70",
                    "value": 0.000227,
                },
            ]

            add_computation_scenario_id(parameters1, _computation_scenario1.id)
            add_computation_scenario_id(parameters2, _computation_scenario2.id)
            ComputationScenarioParameter.insert_many(
                parameters1 + parameters2
            ).execute()
            transaction.commit()

        # 1. Define test data.
        _importer = PipingImporter()

        # 2. Run test
        with pytest.raises(ValueError) as exception_error:
            _importer.import_orm(_piping_per_section)

        # Assert
        assert str(exception_error.value) == "key not defined for first scenario: d_wvp"

    def test_import_orm_without_model_raises_value_error(self):
        # Setup
        _importer = PipingImporter()
        _expected_mssg = "No valid value given for MechanismPerSection."

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # Assert
        assert str(value_error.value) == _expected_mssg
