import numpy as np
import pandas as pd

from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.parameter import Parameter


class StabilityInnerSimpleImporter(OrmImporterProtocol):
    def _set_parameters(
        self, input: MechanismInput, parameters: list[Parameter]
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

        mechanism_input = MechanismInput("StabilityInner")
        self._set_parameters(mechanism_input, orm_model.parameters.select())

        return mechanism_input
