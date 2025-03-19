import numpy as np
import pandas as pd

import vrtool.probabilistic_tools.probabilistic_functions as pb_functions
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.flood_defence_system.failure_mechanism_collection import (
    FailureMechanismCollection,
)

BETA_THRESHOLD: float = 8.0


# Class describing safety assessments of a section:
class SectionReliability:
    load: LoadInput
    failure_mechanisms: FailureMechanismCollection
    # Result stored during calculate_section_reliability
    SectionReliability: pd.array

    def __init__(self) -> None:
        self.failure_mechanisms = FailureMechanismCollection()

    def _get_upscale_cross_sectional_probability(self, section_length: float, mechanism_pf: float, mechanism_a: float, mechanism_b: float) -> float:
        # N = a * L_section / b
        _n_value = mechanism_a * section_length / mechanism_b
        return min(
            1 - (1 - mechanism_pf) ** _n_value, 1.0 / 2
        )

    def calculate_section_reliability(self, dike_section_props: dict[str, float]):
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
                _pf = _mechanism_collection.Reliability[str(_range_val)].Pf
                if mechanism in [MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT]:
                    _pf_mechanisms_time[_count, _range_idx] = _pf
                elif mechanism in [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING]:
                    # underneath one can choose whether to upscale within sections or not:
                    # N = 1
                    _mechanism_a = dike_section_props["a_piping"] if mechanism is MechanismEnum.PIPING else dike_section_props["a_stability_inner"]
                    _mechanism_b = dike_section_props["sf_piping"] if mechanism is MechanismEnum.PIPING else dike_section_props["sf_stability_inner"]
                    _pf_mechanisms_time[_count, _range_idx] = self._get_upscale_cross_sectional_probability(dike_section_props["length"], _pf, _mechanism_a, _mechanism_b)

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

        # replace values greater than the threshold with the threshold itself.
        self.SectionReliability[
            self.SectionReliability > BETA_THRESHOLD
        ] = BETA_THRESHOLD
