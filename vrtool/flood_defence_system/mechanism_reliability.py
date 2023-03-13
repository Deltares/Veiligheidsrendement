from typing import Optional

from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.stability_inner import (
    StabilityInnerSimpleInput,
    StabilityInnerSimpleCalculator,
)
from vrtool.failure_mechanisms.general import (
    GenericFailureMechanismInput,
    GenericFailureMechanismCalculator,
)

from vrtool.failure_mechanisms.overflow import (
    OverflowSimpleInput,
    OverflowHydraRingInput,
    OverflowHydraRingCalculator,
    OverflowSimpleCalculator,
)
from vrtool.failure_mechanisms.piping import PipingSemiProbabilisticCalculator

from vrtool.flood_defence_system.load_input import LoadInput


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
        mechanism: str,
        year: float,
        traject_info: dict,
        strength: Optional[MechanismInput],
        load: Optional[LoadInput],
    ):
        # This routine calculates cross-sectional reliability indices based on different types of calculations.
        if self.type == "DirectInput":
            self.beta, self.Pf = self._calculate_direct_input(strength, year)

        if self.type == "HRING":
            if mechanism == "Overflow":
                self.beta, self.Pf = self._calculate_hydra_ring_overflow(
                    self.Input, year, self.t_0
                )
            else:
                raise Exception(
                    "Unknown computation type HRING for {}".format(mechanism)
                )
        if self.type == "Simple":
            if mechanism == "StabilityInner":
                self.beta, self.Pf = self._calculate_simple_stability_inner(
                    strength, year
                )
            elif mechanism == "Overflow":  # specific for SAFE
                self.beta, self.Pf = self._calculate_simple_overflow(
                    strength, year, load
                )
            elif mechanism == "Piping":
                pass
        elif self.type == "SemiProb":
            # semi probabilistic assessment, only available for piping
            if mechanism == "Piping":
                calculator = PipingSemiProbabilisticCalculator(
                    strength, load, self.t_0, traject_info
                )
                self.Beta, self.Pf = calculator.calculate(year)

    def _calculate_direct_input(
        self, mechanism_input: MechanismInput, year: int
    ) -> tuple[float, float]:
        _mechanism_input = GenericFailureMechanismInput.from_mechanism_input(
            mechanism_input
        )
        calculator = GenericFailureMechanismCalculator(_mechanism_input)
        return calculator.calculate(year)

    def _calculate_simple_stability_inner(
        self, mechanism_input: MechanismInput, year: int
    ) -> tuple[float, float]:
        _mechanism_input = StabilityInnerSimpleInput.from_mechanism_input(
            mechanism_input
        )
        calculator = StabilityInnerSimpleCalculator(_mechanism_input)
        return calculator.calculate(year)

    def _calculate_simple_overflow(
        self, mechanism_input: MechanismInput, year: int, load
    ) -> tuple[float, float]:

        _mechanism_input = OverflowSimpleInput.from_mechanism_input(mechanism_input)
        calculator = OverflowSimpleCalculator(_mechanism_input, load)
        return calculator.calculate(year)

    def _calculate_hydra_ring_overflow(
        self, mechanism_input: MechanismInput, year: int, initial_year: int
    ):
        _mechanism_input = OverflowHydraRingInput.from_mechanism_input(mechanism_input)
        calculator = OverflowHydraRingCalculator(_mechanism_input, initial_year)

        return calculator.calculate(year)
