import numpy as np
import pandas as pd

import vrtool.probabilistic_tools.probabilistic_functions as pb_functions
from vrtool.common.enums.mechanism_enum import MechanismEnum
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
        _available_mechanisms = self.failure_mechanisms.get_available_mechanisms()
        _calculation_years = self.failure_mechanisms.get_calculation_years()

        _t_range = [int(_year_str) for _year_str in _calculation_years]
        _pf_mechanisms_time = np.zeros((len(_available_mechanisms), len(_t_range)))
        _count = 0
        for mechanism in _available_mechanisms:  # mechanisms
            for _range_idx, _range_val in enumerate(_t_range):
                _mechanism_collection = (
                    self.failure_mechanisms.get_mechanism_reliability_collection(
                        mechanism
                    )
                )

                if mechanism in [MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT]:
                    _pf_mechanisms_time[_count, _range_idx] = (
                        _mechanism_collection.Reliability[str(_range_val)].Pf
                    )
                elif mechanism in [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING]:
                    pf = _mechanism_collection.Reliability[str(_range_val)].Pf
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
                    _pf_mechanisms_time[_count, _range_idx] = min(
                        1 - (1 - pf) ** N, 1.0 / 2
                    )
            _count += 1

        # Do we want beta or failure probability? Preferably beta as output
        _beta_mech_time = pd.DataFrame(
            pb_functions.pf_to_beta(_pf_mechanisms_time),
            columns=_calculation_years,
            index=list(map(str, _available_mechanisms)),
        )
        _beta_time = pd.DataFrame(
            [pb_functions.pf_to_beta(np.sum(_pf_mechanisms_time, axis=0))],
            columns=_calculation_years,
            index=["Section"],
        )
        self.SectionReliability = pd.concat((_beta_mech_time, _beta_time))
        # TODO add output as probability so we dont have to switch using scipystats all the time.
