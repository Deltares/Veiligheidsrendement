import numpy as np

from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.parameter import Parameter


class PipingImporter(OrmImporterProtocol):
    def _set_parameters(
        self,
        input: MechanismInput,
        parameters: list[Parameter],
        index: int,
        nrScenarios: int,
    ) -> None:
        for parameter in parameters:
            key = parameter.parameter
            if index == 0:
                input.input[key] = np.zeros(nrScenarios)
                if key[-3:] == "(t)":
                    input.temporals.append(key)

            if key in input.input:
                input.input[key][index] = parameter.value
            else:
                # note that we do not check on the presence of all keys
                # except for the first scenario; if it is missing the value stays at 0.0
                raise ValueError("key not defined for first scenario: " + key)

    def import_orm(self, orm_model: MechanismPerSection) -> MechanismInput:
        if not orm_model:
            raise ValueError(
                f"No valid value given for {MechanismPerSection.__name__}."
            )

        mechanism_input = MechanismInput("Piping")
        mechanism_input.temporals = []

        index = 0

        computation_scenarios: list[
            ComputationScenario
        ] = orm_model.computation_scenarios.select()
        nr_of_scenarios = len(computation_scenarios)
        scenario_probablity_key = "P_scenario"
        mechanism_input.input[scenario_probablity_key] = np.zeros(nr_of_scenarios)
        _scenario_key = "Scenario"
        mechanism_input.input[_scenario_key] = []
        for _c_scenario in computation_scenarios:
            self._set_parameters(
                mechanism_input, _c_scenario.parameters.select(), index, nr_of_scenarios
            )
            mechanism_input.input[_scenario_key].append(_c_scenario.computation_name)
            mechanism_input.input[scenario_probablity_key][
                index
            ] = _c_scenario.scenario_probability
            index += 1

        return mechanism_input
