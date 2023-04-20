from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from geolib import DStabilityModel

from vrtool.failure_mechanisms.mechanism_input import MechanismInput


class StabilityInnerDStabilityInput:

    @staticmethod
    def from_stix_input(
        mechanism_input: MechanismInput,
        externals_directory: Path
    ) -> float:
        """Create a StabilityInnerDStabilityInput from a mechanism input.

        Args:
                mechanism_input (MechanismInput): The mechanism input.
                rerun_stix (bool, optional): Whether to rerun the stix. Defaults to False.
                stage_id_result (Optional[int], optional): The stage id for which result is fetched from. Defaults to None,
                in that case the last stage is considered. This is the stage id in the stix file, not necessarily the order
                of the stage in the GUI!

        Returns:
                float: The calculated safety factor.
        """
        from vrtool.failure_mechanisms.stability_inner.dstability_wrapper import DStabilityWrapper

        _wrapper = DStabilityWrapper(Path(mechanism_input.input.get("STIXNAAM", "")), externals_directory)
        if mechanism_input.input.get("RERUN_STIX"):
            _wrapper.rerun_stix()

        return np.array(_wrapper.get_safety_factor(mechanism_input.input.get("STAGE_ID_RESULT", None)))
