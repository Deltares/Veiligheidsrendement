from typing import Callable

import numpy as np
import pytest
from peewee import SqliteDatabase

from tests.orm.io import add_computation_scenario_id
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.mechanism_simple_importer import MechanismSimpleImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_scenario_parameter import (
    ComputationScenarioParameter,
)


class TestMechanismSimpleImporter:
    def test_initialize(self):
        # Call
        _importer = MechanismSimpleImporter()

        # Assert
        assert isinstance(_importer, MechanismSimpleImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(
        self,
        empty_db_context: SqliteDatabase,
        get_basic_computation_scenario: Callable[[], ComputationScenario],
    ):
        # Setup
        parameters = [
            {
                "parameter": "sf_2025",
                "value": 0.9719626168224299,
            },
            {
                "parameter": "sf_2075",
                "value": 0.9626168224299065,
            },
            {
                "parameter": "d_cover",
                "value": 3.778095454216003,
            },
            {
                "parameter": "dsf/dberm",
                "value": 0.02,
            },
            {
                "parameter": "beta_2025",
                "value": 3.322508516027601,
            },
            {
                "parameter": "beta_2075",
                "value": 3.264279267476053,
            },
            {
                "parameter": "dbeta/dberm",
                "value": 0.1333333333333333,
            },
        ]

        with empty_db_context.atomic() as transaction:
            _computation_scenario = get_basic_computation_scenario()

            add_computation_scenario_id(parameters, _computation_scenario.id)
            ComputationScenarioParameter.insert_many(parameters).execute()

            transaction.commit()

        _importer = MechanismSimpleImporter()

        # Call
        _mechanism_input = _importer.import_orm(
            _computation_scenario.mechanism_per_section
        )

        # Assert
        assert isinstance(_mechanism_input, MechanismInput)

        assert _mechanism_input.mechanism == MechanismEnum.OVERFLOW
        assert len(_mechanism_input.input) == len(parameters) + 3
        assert _mechanism_input.input["Scenario"] == [
            _computation_scenario.scenario_name
        ]
        assert _mechanism_input.input["P_scenario"] == [
            _computation_scenario.scenario_probability
        ]
        assert _mechanism_input.input["Pf"] == [
            _computation_scenario.probability_of_failure
        ]
        for parameter in parameters:
            mechanism_parameter = _mechanism_input.input[parameter.get("parameter")]
            assert isinstance(mechanism_parameter, np.ndarray)
            assert mechanism_parameter == pytest.approx(parameter.get("value"))

    def test_import_orm_without_model_raises_value_error(self):
        # Setup
        _importer = MechanismSimpleImporter()
        _expected_mssg = "No valid value given for MechanismPerSection."

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # Assert
        assert str(value_error.value) == _expected_mssg
