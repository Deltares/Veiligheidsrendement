import numpy as np
import pandas as pd

import vrtool.probabilistic_tools.probabilistic_functions as pb_functions
from vrtool.common.enums import MechanismEnum
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.flood_defence_system.failure_mechanism_collection import (
    FailureMechanismCollection,
)


# Class describing safety assessments of a section:
class SectionReliability:
    load: LoadInput
    failure_mechanisms: FailureMechanismCollection

    def __init__(self) -> None:
        self.failure_mechanisms = FailureMechanismCollection()

    def calculate_section_reliability(self):
        # This routine translates cross-sectional to section reliability indices

        # TODO Add optional interpolation here.
        available_mechanisms = self.failure_mechanisms.get_available_mechanisms()
        calculation_years = self.failure_mechanisms.get_calculation_years()

        trange = [int(i) for i in calculation_years]
        pf_mechanisms_time = np.zeros((len(available_mechanisms), len(trange)))
        count = 0
        for mechanism in available_mechanisms:  # mechanisms
            for j in range(0, len(trange)):
                mechanism_collection = (
                    self.failure_mechanisms.get_mechanism_reliability_collection(
                        mechanism
                    )
                )

                if mechanism in [MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT]:
                    pf_mechanisms_time[count, j] = mechanism_collection.Reliability[
                        str(trange[j])
                    ].Pf
                elif mechanism in [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING]:
                    pf = mechanism_collection.Reliability[str(trange[j])].Pf
                    # underneath one can choose whether to upscale within sections or not:
                    N = 1
                    # For StabilityInner:
                    # N = length/TrajectInfo['bStabilityInner']
                    # N = TrajectInfo['aStabilityInner']*length/TrajectInfo['bStabilityInner']
                    #
                    # For Piping:
                    # N = length/TrajectInfo['bPiping']
                    # N = TrajectInfo['aPiping'] * length / TrajectInfo['bPiping']
                    # pf_mechanisms_time[count, j] = min(1 - (1 - pf) ** N,1./100)

                    # pf_mechanisms_time[count,j] = min(1 - (1 - pf) ** N,1./100)
                    pf_mechanisms_time[count, j] = min(1 - (1 - pf) ** N, 1.0 / 2)
            count += 1

        # Do we want beta or failure probability? Preferably beta as output
        beta_mech_time = pd.DataFrame(
            pb_functions.pf_to_beta(pf_mechanisms_time),
            columns=calculation_years,
            index=list(_mech.name for _mech in available_mechanisms),
        )
        beta_time = pd.DataFrame(
            [pb_functions.pf_to_beta(np.sum(pf_mechanisms_time, axis=0))],
            columns=calculation_years,
            index=["Section"],
        )
        self.SectionReliability = pd.concat((beta_mech_time, beta_time))
        # TODO add output as probability so we dont have to switch using scipystats all the time.
