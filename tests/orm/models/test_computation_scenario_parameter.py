from typing import Callable

from tests.orm import with_empty_db_context
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_scenario_parameter import (
    ComputationScenarioParameter,
)
from vrtool.orm.models.orm_base_model import OrmBaseModel


class TestComputationScenarioParameter:
    @with_empty_db_context
    def test_initialize_with_database_fixture(
        self, get_basic_computation_scenario: Callable[[], ComputationScenario]
    ):
        # 1. Define test data.
        _computation_scenario = get_basic_computation_scenario()
        assert isinstance(_computation_scenario, ComputationScenario)

        # 2. Run test.
        _parameter = ComputationScenarioParameter.create(
            computation_scenario=_computation_scenario,
            parameter="TestParameter",
            value=4.2,
        )

        # 3. Verify expectations
        assert isinstance(_parameter, ComputationScenarioParameter)
        assert isinstance(_parameter, OrmBaseModel)
        assert _parameter.computation_scenario == _computation_scenario
        assert _parameter.parameter == "TestParameter"
        assert _parameter.value == 4.2

        assert _parameter in _computation_scenario.computation_scenario_parameters

    @with_empty_db_context
    def test_initialize_two_parameters_with_same_name_with_database_fixture(
        self, get_basic_computation_scenario: Callable[[], ComputationScenario]
    ):
        # 1. Define test data.
        _scenario_a = get_basic_computation_scenario()
        assert isinstance(_scenario_a, ComputationScenario)

        _scenario_b = ComputationScenario.create(
            mechanism_per_section=_scenario_a.mechanism_per_section,
            computation_type=_scenario_a.computation_type,
            computation_name=_scenario_a.computation_name,
            scenario_name=_scenario_a.scenario_name,
            scenario_probability=0.24,
            probability_of_failure=0.24,
        )

        # 2. Run test.
        _parameter_a = ComputationScenarioParameter.create(
            computation_scenario=_scenario_a, parameter="TestParameter", value=4.2
        )
        _parameter_b = ComputationScenarioParameter.create(
            computation_scenario=_scenario_b, parameter="TestParameter", value=2.4
        )

        # 3. Verify expectations
        assert _parameter_a in _scenario_a.computation_scenario_parameters
        assert _parameter_b in _scenario_b.computation_scenario_parameters

    @with_empty_db_context
    def test_on_delete_computation_scenario_cascades(
        self, get_basic_computation_scenario: Callable[[], ComputationScenario]
    ):
        # 1. Define test data.
        _computation_scenario = get_basic_computation_scenario()
        assert not any(ComputationScenarioParameter.select())

        ComputationScenarioParameter.create(
            computation_scenario=_computation_scenario,
            parameter="TestParameter",
            value=4.2,
        )
        assert any(ComputationScenarioParameter.select())

        # 2. Run test.
        _computation_scenario.delete_instance()

        # 3. Verify expectations.
        assert not any(ComputationScenarioParameter.select())
