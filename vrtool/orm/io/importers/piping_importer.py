from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.parameter import Parameter


class PipingImporter(OrmImporterProtocol):
    def _set_parameters(
        self, input: MechanismInput, parameters: list[Parameter]
    ) -> None:
        for parameter in parameters:
            input.input[parameter.parameter] = parameter.value

    def import_orm(self, orm_model: ComputationScenario) -> MechanismInput:
        mechanism_input = MechanismInput("Piping")

        self._set_parameters(mechanism_input, orm_model.parameters.select())

        return mechanism_input
