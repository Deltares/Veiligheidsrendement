import numpy as np
import pytest
from peewee import SqliteDatabase

from tests.orm import empty_db_fixture, get_basic_computation_scenario
from tests.orm.io import add_computation_scenario_id
from vrtool.common.enums import MechanismEnum
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.stability_inner_simple_importer import (
    StabilityInnerSimpleImporter,
)
from vrtool.orm.models.computation_scenario_parameter import (
    ComputationScenarioParameter,
)


class TestStabilityInnerSimpleImporter:
    def test_initialize(self):
        # Call
        _importer = StabilityInnerSimpleImporter()

        # Assert
        assert isinstance(_importer, StabilityInnerSimpleImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(self, empty_db_fixture: SqliteDatabase):
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

        with empty_db_fixture.atomic() as transaction:
            _computation_scenario = get_basic_computation_scenario()

            add_computation_scenario_id(parameters, _computation_scenario.id)
            ComputationScenarioParameter.insert_many(parameters).execute()

            transaction.commit()

        _importer = StabilityInnerSimpleImporter()

        # Call
        _mechanism_input = _importer.import_orm(_computation_scenario)

        # Assert
        assert isinstance(_mechanism_input, MechanismInput)

        assert _mechanism_input.mechanism == MechanismEnum.STABILITY_INNER
        assert len(_mechanism_input.input) == len(parameters)
        for parameter in parameters:
            mechanism_parameter = _mechanism_input.input[parameter.get("parameter")]
            assert isinstance(mechanism_parameter, np.ndarray)
            assert mechanism_parameter == pytest.approx(parameter.get("value"))

    def test_import_orm_without_model_raises_value_error(self):
        # Setup
        _importer = StabilityInnerSimpleImporter()
        _expected_mssg = "No valid value given for ComputationScenario."

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # Assert
        assert str(value_error.value) == _expected_mssg
