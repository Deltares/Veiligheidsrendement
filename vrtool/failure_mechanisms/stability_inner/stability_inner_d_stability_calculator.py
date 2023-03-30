from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from geolib import DStabilityModel

from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.stability_inner.stability_inner_functions import calculate_reliability
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


# TODO TO put in another file



@dataclass
class StabilityInnerDStabilityInput:
    safety_factor: np.ndarray

    @classmethod
    def from_stix_input(
            cls,
            mechanism_input: MechanismInput,
    ):
        _stix_path = Path(mechanism_input.input.get("STIXNAAM", ""))
        _dstability_model = DStabilityModel()
        _dstability_model.parse(_stix_path)

        ## the tested stix only had one stage for which results were stored, so calling dm.output works in this case
        # but this might not be robust enough. Current version of GEOLIB (0.4.0) does not handle multistage outputs.
        # Put this one on hold for on until GEOLIB is updated

        try:
            _safety_factor = _dstability_model.output.FactorOfSafety
        except ValueError:  # TODO ideally if no output, rerun the stix
            raise Exception(f"No output found in the provided stix {_stix_path.parts[-1]}")

        _input = cls(
            safety_factor=_safety_factor
        )
        return _input


class StabilityInnerDStabilityCalculator(FailureMechanismCalculatorProtocol):
    """
    Contains all methods related to performing a stability inner calculation.
    """

    def __init__(self, mechanism_input: StabilityInnerDStabilityInput) -> None:
        self._mechanism_input = mechanism_input

    def calculate(self, year: int) -> Tuple[float, float]:
        """Calculate the reliability and the probability of failure from an input safety factor of a stix file.
        The reliability is insensitive to the year.
        """
        beta = np.min([self._mechanism_input.safety_factor, 8.0])

        return beta, beta_to_pf(beta)
