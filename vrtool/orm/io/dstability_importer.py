from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.parameter import Parameter


class DStabilityImporter(OrmImporterProtocol):
    def _set_parameters(
        self, input: MechanismInput, parameters: list[Parameter]
    ) -> None:
        for parameter in parameters:
            input.input[parameter.parameter] = parameter.value

    def import_orm(self, orm_model: ComputationScenario) -> MechanismInput:
        if not orm_model:
            raise ValueError(
                f"No valid value given for {ComputationScenario.__name__}."
            )

        mechanism_input = MechanismInput("StabilityInner")

        self._set_parameters(mechanism_input, orm_model.parameters.select())

        supporting_files = orm_model.supporting_files.select()

        if len(supporting_files) != 1:
            raise ValueError("Invalid number of stix files.")

        mechanism_input.input["stix_file"] = supporting_files.get().filename

        return mechanism_input
