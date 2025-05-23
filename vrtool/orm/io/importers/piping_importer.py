import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_scenario_parameter import (
    ComputationScenarioParameter,
)
from vrtool.orm.models.mechanism_per_section import MechanismPerSection


class PipingImporter(OrmImporterProtocol):
    def _set_parameters(
        self,
        mechanism_input: MechanismInput,
        parameters: list[ComputationScenarioParameter],
        index: int,
        scenarios_length: int,
    ) -> None:
        for parameter in parameters:
            key = parameter.parameter
            if index == 0:
                mechanism_input.input[key] = np.zeros(scenarios_length)
                if key[-3:] == "(t)":
                    mechanism_input.temporals.append(key)

            if key in mechanism_input.input:
                mechanism_input.input[key][index] = parameter.value
            else:
                # note that we do not check on the presence of all keys
                # except for the first scenario; if it is missing the value stays at 0.0
                raise ValueError("key not defined for first scenario: " + key)

    def import_orm(self, orm_model: MechanismPerSection) -> MechanismInput:
        if not orm_model:
            raise ValueError(
                f"No valid value given for {MechanismPerSection.__name__}."
            )

        mechanism_input = MechanismInput(MechanismEnum.PIPING)
        mechanism_input.temporals = []

        index = 0

        computation_scenarios: list[
            ComputationScenario
        ] = orm_model.computation_scenarios.select()
        nr_of_scenarios = len(computation_scenarios)
        scenario_probablity_key = "P_scenario"
        mechanism_input.input[scenario_probablity_key] = np.zeros(nr_of_scenarios)
        mechanism_input.input["Beta"] = np.zeros(nr_of_scenarios)
        _scenario_key = "Scenario"
        mechanism_input.input[_scenario_key] = []
        for _c_scenario in computation_scenarios:
            self._set_parameters(
                mechanism_input,
                _c_scenario.computation_scenario_parameters.select(),
                index,
                nr_of_scenarios,
            )
            mechanism_input.input[_scenario_key].append(_c_scenario.scenario_name)
            mechanism_input.input[scenario_probablity_key][
                index
            ] = _c_scenario.scenario_probability
            # TODO: VRTOOL-340. This does not support multiple scenarios.
            mechanism_input.input["Pf"] = _c_scenario.probability_of_failure
            index += 1

        return mechanism_input
