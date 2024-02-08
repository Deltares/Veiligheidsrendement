import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_scenario_parameter import (
    ComputationScenarioParameter,
)
from vrtool.orm.models.mechanism_per_section import MechanismPerSection


class StabilityInnerSimpleImporter(OrmImporterProtocol):
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

        mechanism_input = MechanismInput(MechanismEnum.STABILITY_INNER)
        for _computation_scenario in orm_model.computation_scenarios:
            self._set_parameters(
                mechanism_input,
                _computation_scenario.computation_scenario_parameters.select(),
            )

        return mechanism_input
