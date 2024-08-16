from pathlib import Path

import numpy as np

from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_scenario_parameter import (
    ComputationScenarioParameter,
)
from vrtool.orm.models.mechanism_per_section import MechanismPerSection


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

        self._computation_type = ComputationTypeEnum.DSTABILITY

    def _set_parameters(
        self, mech_input: MechanismInput, parameters: list[ComputationScenarioParameter]
    ) -> None:
        for parameter in parameters:
            _key = parameter.parameter.lower().strip()
            if _key not in mech_input.input.keys():
                mech_input.input[_key] = np.array([float(parameter.value)])
            else:
                mech_input.input[_key] = np.append(
                    mech_input.input[_key], float(parameter.value)
                )

    def import_orm(self, orm_model: MechanismPerSection) -> MechanismInput:
        if not orm_model:
            raise ValueError(
                f"No valid value given for {MechanismPerSection.__name__}."
            )

        if any(
            cs.computation_type.name != self._computation_type.legacy_name
            for cs in orm_model.computation_scenarios
        ):
            raise ValueError(
                f"All computation types must be '{self._computation_type}'."
            )

        mechanism_input = MechanismInput(MechanismEnum.STABILITY_INNER)

        for _computation_scenario in orm_model.computation_scenarios:
            self._set_parameters(
                mechanism_input,
                _computation_scenario.computation_scenario_parameters.select(),
            )

        # TO DO: How are we supporting multiple scenarios with dstability files?
        supporting_files = (
            orm_model.computation_scenarios.select().get().supporting_files.select()
        )

        if len(supporting_files) != 1:
            raise ValueError("Invalid number of stix files.")

        mechanism_input.input["STIXNAAM"] = (
            self._stix_directory / supporting_files.get().filename
        )
        mechanism_input.input["DStability_exe_path"] = str(
            self._dstability_exe_directory
        )

        return mechanism_input
