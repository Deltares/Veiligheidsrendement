import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_scenario_parameter import (
    ComputationScenarioParameter,
)


class StabilityInnerSimpleImporter(OrmImporterProtocol):
    def _set_parameters(
        self, input: MechanismInput, parameters: list[ComputationScenarioParameter]
    ) -> None:
        for parameter in parameters:
            input.input[parameter.parameter.lower().strip()] = np.array(
                [float(parameter.value)]
            )

    def import_orm(self, orm_model: ComputationScenario) -> MechanismInput:
        if not orm_model:
            raise ValueError(
                f"No valid value given for {ComputationScenario.__name__}."
            )

        mechanism_input = MechanismInput(MechanismEnum.STABILITY_INNER)
        self._set_parameters(
            mechanism_input, orm_model.computation_scenario_parameters.select()
        )

        return mechanism_input
