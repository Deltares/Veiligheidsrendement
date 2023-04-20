from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from geolib import DStabilityModel

from vrtool.failure_mechanisms.mechanism_input import MechanismInput


@dataclass
class StabilityInnerDStabilityInput:
    safety_factor: np.ndarray

    @classmethod
    def from_stix_input(
        cls,
        mechanism_input: MechanismInput,
    ):
        """Create a StabilityInnerDStabilityInput from a mechanism input.

        Args:
                mechanism_input (MechanismInput): The mechanism input.
                rerun_stix (bool, optional): Whether to rerun the stix. Defaults to False.
                stage_id_result (Optional[int], optional): The stage id for which result is fetched from. Defaults to None,
                in that case the last stage is considered. This is the stage id in the stix file, not necessarily the order
                of the stage in the GUI!

        Returns:
                StabilityInnerDStabilityInput: The StabilityInnerDStabilityInput.
        """
        _stix_path = Path(mechanism_input.input.get("STIXNAAM", ""))
        _stage_id_result = mechanism_input.input.get("STAGE_ID_RESULT", None)
        _dstability_model = DStabilityModel()
        _dstability_model.parse(_stix_path)

        if mechanism_input.input.get("RERUN_STIX"):
            _dstability_model.meta.console_folder = (
                Path(__file__).parent.parent.parent.parent
                / "externals/D-Stability 2022.01.2/bin"
            )
            _dstability_model.execute()

        if _stage_id_result is None:
            _safety_factor = _dstability_model.output[-1].FactorOfSafety

        else:
            _result_id = _dstability_model.datastructure.stages[
                _stage_id_result
            ].ResultId
            if _result_id is None:
                raise Exception(
                    f"The requested stage id {_result_id} does not have saved results in the provided stix {_stix_path.parts[-1]}, please rerun DStability"
                )

            for stage_output in _dstability_model.output:
                if stage_output is not None and stage_output.Id == _result_id:

                    _safety_factor = stage_output.FactorOfSafety
                    break  # Factor of Safety is found, no need to continue iterating

        _input = cls(safety_factor=np.array(_safety_factor))
        return _input
