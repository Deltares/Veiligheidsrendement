import numpy as np
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.parameter import Parameter


class PipingImporter(OrmImporterProtocol):
    def _set_parameters(
        self, input: MechanismInput, parameters: list[Parameter], isFirst: bool
    ) -> None:
        for parameter in parameters:
            if isFirst:
                input.input[parameter.parameter] = np.array([parameter.value])
            else:
                input.input[parameter.parameter] = np.append(
                    input.input[parameter.parameter], parameter.value
                )

    def import_orm(self, orm_model: list[ComputationScenario]) -> MechanismInput:
        mechanism_input = MechanismInput("Piping")

        isFirst = True
        for scenario in orm_model:
            self._set_parameters(mechanism_input, scenario.parameters.select(), isFirst)
            isFirst = False

        return mechanism_input
