from pathlib import Path
from typing import Optional

import numpy as np

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol
from vrtool.failure_mechanisms.general import (
    GenericFailureMechanismCalculator,
    GenericFailureMechanismInput,
)
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.overflow import (
    OverflowHydraRingCalculator,
    OverflowHydraRingInput,
    OverflowSimpleCalculator,
    OverflowSimpleInput,
)
from vrtool.failure_mechanisms.piping import PipingSemiProbabilisticCalculator
from vrtool.failure_mechanisms.stability_inner import (
    StabilityInnerSimpleCalculator,
    StabilityInnerSimpleInput,
)
from vrtool.failure_mechanisms.stability_inner.dstability_wrapper import DStabilityWrapper
from vrtool.failure_mechanisms.stability_inner.stability_inner_d_stability_calculator import (
    StabilityInnerDStabilityCalculator
)

class MechanismReliability:
    # This class contains evaluations of the reliability for a mechanism in a given year.
    def __init__(self, mechanism, type, t_0: int, copy_or_calculate="calculate"):
        # Initialize: set mechanism and type. These are the most important basic parameters
        self.mechanism = mechanism
        self.type = type
        self.t_0 = t_0
        self.copy_or_calculate = copy_or_calculate

        self.Input = MechanismInput(self.mechanism)
        if mechanism == "Piping":
            self.gamma_schem_heave = 1  # 1.05
            self.gamma_schem_upl = 1  # 1.05
            self.gamma_schem_pip = 1  # 1.05
        else:
            pass

    def __clearvalues__(self):
        # clear all values
        keys = self.__dict__.keys()
        for i in keys:
            # if i is not 'mechanism':
            if i != "mechanism":
                setattr(self, i, None)

    def calcReliability(
        self,
        strength: MechanismInput,
        load: LoadInput,
        mechanism: str,
        year: float,
        traject_info: DikeTrajectInfo,
    ):
        calculator = self._get_failure_mechanism_calculator(
            mechanism, traject_info, strength, load
        )

        self.Beta, self.Pf = calculator.calculate(year)

    def _get_failure_mechanism_calculator(
        self,
        mechanism: str,
        traject_info: DikeTrajectInfo,
        strength: Optional[MechanismInput],
        load: Optional[LoadInput],
    ) -> FailureMechanismCalculatorProtocol:

        if self.type == "DirectInput":
            return self._get_direct_input_calculator(strength)

        if self.type == "HRING":
            return self._get_hydra_ring_calculator(mechanism, self.Input)

        if self.type == "Simple":
            return self._get_simple_calculator(mechanism, strength, load)

        if self.type == "SemiProb":
            return self._get_semi_probabilistic_calculator(
                mechanism, strength, load, traject_info
            )
        if self.type == "DStability":
            return self._get_d_stability_calculator(mechanism, strength)

        raise Exception("Unknown computation type {}".format(self.type))

    def _get_direct_input_calculator(
        self, mechanism_input: MechanismInput
    ) -> FailureMechanismCalculatorProtocol:
        _mechanism_input = GenericFailureMechanismInput.from_mechanism_input(
            mechanism_input
        )
        return GenericFailureMechanismCalculator(_mechanism_input)

    def _get_hydra_ring_calculator(
        self, mechanism: str, mechanism_input: MechanismInput
    ) -> FailureMechanismCalculatorProtocol:
        if mechanism == "Overflow":
            _mechanism_input = OverflowHydraRingInput.from_mechanism_input(
                mechanism_input
            )
            return OverflowHydraRingCalculator(_mechanism_input, self.t_0)

        raise Exception("Unknown computation type HRING for {}".format(mechanism))

    def _get_simple_calculator(
        self, mechanism, mechanism_input: MechanismInput, load: LoadInput
    ) -> FailureMechanismCalculatorProtocol:
        if mechanism == "StabilityInner":
            _mechanism_input = StabilityInnerSimpleInput.from_mechanism_input(
                mechanism_input
            )
            return StabilityInnerSimpleCalculator(_mechanism_input)

        if mechanism == "Overflow":  # specific for SAFE
            _mechanism_input = OverflowSimpleInput.from_mechanism_input(mechanism_input)
            return OverflowSimpleCalculator(_mechanism_input, load)

        raise Exception("Unknown computation type Simple for {}".format(mechanism))

    def _get_semi_probabilistic_calculator(
        self,
        mechanism: str,
        mechanism_input: MechanismInput,
        load: LoadInput,
        traject_info: DikeTrajectInfo,
    ) -> FailureMechanismCalculatorProtocol:
        if mechanism == "Piping":
            return PipingSemiProbabilisticCalculator(
                mechanism_input, load, self.t_0, traject_info
            )

        raise Exception("Unknown computation type SemiProb for {}".format(mechanism))

    def _get_d_stability_calculator(
        self,
        mechanism: str,
        mechanism_input: MechanismInput,
    ) -> FailureMechanismCalculatorProtocol:

        if mechanism == "StabilityInner":

            _wrapper = DStabilityWrapper(stix_path=Path(mechanism_input.input.get("STIXNAAM", "")),
                                         externals_path=Path(mechanism_input.input.get("DStability_exe_path")))
            if mechanism_input.input.get("RERUN_STIX"):
                _wrapper.rerun_stix()
            _mechanism_input = np.array(_wrapper.get_safety_factor(int(mechanism_input.input.get("STAGEID")[0])))
            return StabilityInnerDStabilityCalculator(_mechanism_input)

        raise Exception("Unknown computation type DStability for {}".format(mechanism))
