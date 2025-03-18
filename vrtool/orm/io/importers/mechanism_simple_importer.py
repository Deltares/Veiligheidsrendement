import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario_parameter import (
    ComputationScenarioParameter,
)
from vrtool.orm.models.mechanism_per_section import MechanismPerSection


class MechanismSimpleImporter(OrmImporterProtocol):  
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

        mechanism_input = MechanismInput(mechanism=MechanismEnum.get_enum(orm_model.mechanism.name))
        _scenario_key = "Scenario"
        _scenario_probablity_key = "P_scenario"
        _probability_of_failure = "Pf"

        mechanism_input.input[_scenario_key] = []
        mechanism_input.input[_scenario_probablity_key] = np.array([])
        mechanism_input.input[_probability_of_failure] = np.array([])

        def _append_to_numpy_input(input_key: str, value: float) -> None:
            mechanism_input.input[input_key] = np.append(
                mechanism_input.input[input_key], value
            )

        for _c_scenario in orm_model.computation_scenarios:
            self._set_parameters(
                mechanism_input,
                _c_scenario.computation_scenario_parameters.select(),
            )
            mechanism_input.input[_scenario_key].append(_c_scenario.scenario_name)
            _append_to_numpy_input(
                _scenario_probablity_key, _c_scenario.scenario_probability
            )
            _append_to_numpy_input(
                _probability_of_failure, _c_scenario.probability_of_failure
            )

        return mechanism_input
