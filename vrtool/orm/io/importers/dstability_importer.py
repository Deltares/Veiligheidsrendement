from pathlib import Path

from vrtool.common.enums import MechanismEnum
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_scenario_parameter import (
    ComputationScenarioParameter,
)


class DStabilityImporter(OrmImporterProtocol):
    def __init__(self, externals_directory: Path, stix_directory: Path) -> None:
        """
        Creates an instance of a DStability importer.

        Args:
            externals_directory (Path): The directory in which the externals are located.
            stix_directory (Path): The directory in which the stix files are located.
        """
        self._dstability_exe_directory = externals_directory
        self._stix_directory = stix_directory

        self._computation_type = "DSTABILITY"

    def _set_parameters(
        self, input: MechanismInput, parameters: list[ComputationScenarioParameter]
    ) -> None:
        for parameter in parameters:
            input.input[parameter.parameter] = parameter.value

    def import_orm(self, orm_model: ComputationScenario) -> MechanismInput:
        if not orm_model:
            raise ValueError(
                f"No valid value given for {ComputationScenario.__name__}."
            )

        if orm_model.computation_type.name != self._computation_type:
            raise ValueError(f"Computation type must be '{self._computation_type}'.")

        mechanism_input = MechanismInput(MechanismEnum.STABILITY_INNER)

        self._set_parameters(
            mechanism_input, orm_model.computation_scenario_parameters.select()
        )

        supporting_files = orm_model.supporting_files.select()

        if len(supporting_files) != 1:
            raise ValueError("Invalid number of stix files.")

        mechanism_input.input["STIXNAAM"] = (
            self._stix_directory / supporting_files.get().filename
        )
        mechanism_input.input["DStability_exe_path"] = str(
            self._dstability_exe_directory
        )

        return mechanism_input
