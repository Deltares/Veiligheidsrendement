import numpy as np
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
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

            if key in input.input:
                input.input[key][index] = parameter.value
            else:
                raise ValueError("key not defined for first scenario: " + key)

    def import_orm(self, orm_model: list[ComputationScenario]) -> MechanismInput:
        mechanism_input = MechanismInput("Piping")

        index = 0
        nrScenarios = len(orm_model)
        for scenario in orm_model:
            self._set_parameters(
                mechanism_input, scenario.parameters.select(), index, nrScenarios
            )
            index += 1

        return mechanism_input
