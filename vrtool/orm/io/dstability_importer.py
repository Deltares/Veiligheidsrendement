from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.parameter import Parameter
from vrtool.orm.models.supporting_file import SupportingFile


class DStabilityImporter(OrmImporterProtocol):
    def _set_parameters(
        self, input: MechanismInput, parameters: list[Parameter]
    ) -> None:
        for parameter in parameters:
            input.input[parameter.parameter] = parameter.value

    def import_orm(self, orm_model: ComputationScenario) -> MechanismInput:
        mechanism_input = MechanismInput("StabilityInner")

        id = orm_model.select().get().id
        sp = SupportingFile()
        file = (
            sp.select()
            .where(SupportingFile.computation_scenario_id == id)
            .get()
            .filename
        )
        mechanism_input.input["stix_file"] = file
        mechanism_tables = orm_model.mechanism_tables.select()

        return mechanism_input
