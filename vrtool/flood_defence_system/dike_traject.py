from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


class DikeTraject:
    general_info: DikeTrajectInfo
    sections: list[DikeSection]
    probabilities: pd.DataFrame
    mechanisms: list[MechanismEnum]
    t_0: int
    T: list[int]

    # This class contains general information on the dike traject and is used to store all data on the sections
    @classmethod
    def from_config(cls, config: VrtoolConfig) -> DikeTraject:
        """
        Generates a `DikeTraject` from a simple `VrtoolConfig` object.

        Args:
            config (VrtoolConfig): valid `VrtoolConfig` object.

        Raises:
            ValueError: When no traject value has been provided.

        Returns:
            DikeTraject: object containing the related values from the config.
        """
        if not config.traject:
            raise ValueError("No traject given in config.")

        _dike_traject = cls()

        _dike_traject.mechanisms = config.mechanisms
        _dike_traject.t_0 = config.t_0
        _dike_traject.T = config.T

        _dike_traject.sections = DikeSection.get_dike_sections_from_vr_config(config)

        _traject_length = sum(map(lambda x: x.Length, _dike_traject.sections))
        _dike_traject.general_info = DikeTrajectInfo.from_traject_info(
            config.traject, _traject_length
        )

        return _dike_traject

    def set_probabilities(self):
        """routine to make 1 dataframe of all probabilities of a TrajectObject"""
        for i, section in enumerate(self.sections):
            if i == 0:
                _assessment = (
                    section.section_reliability.SectionReliability.reset_index()
                )
                _assessment["Section"] = section.name
                _assessment["Length"] = section.Length
                _assessment.columns = _assessment.columns.astype(str)
                if "mechanism" in _assessment.columns:
                    _assessment = _assessment.rename(columns={"mechanism": "index"})
            else:
                data_to_add = (
                    section.section_reliability.SectionReliability.reset_index()
                )
                data_to_add["Section"] = section.name
                data_to_add["Length"] = section.Length
                data_to_add.columns = data_to_add.columns.astype(str)
                if "mechanism" in data_to_add.columns:
                    data_to_add = data_to_add.rename(columns={"mechanism": "index"})

                _assessment = pd.concat((_assessment, data_to_add))
        _assessment = _assessment.rename(
            columns={"index": "mechanism", "Section": "name"}
        )
        self.probabilities = _assessment.reset_index(drop=True).set_index(
            ["name", "mechanism"]
        )

    def write_initial_assessment_results(
        self,
        case_settings: dict,
    ):
        self.probabilities.to_csv(
            case_settings["directory"].joinpath("InitialAssessment_Betas.csv")
        )


def calc_traject_prob(base, horizon=False, datatype="DataFrame", ts=None, mechs=False):
    pfs = {}
    if horizon:
        trange = np.arange(0, horizon, 1)
    elif ts != None:
        trange = [ts]
    else:
        raise ValueError("No range defined")
    if datatype == "DataFrame":
        ts = base.columns.values
        if not mechs:
            mechs = np.unique(base.index.get_level_values("mechanism").values)
        # mechs = ['OVERFLOW']
    # pf_traject = np.zeros((len(ts),))
    pf_traject = np.zeros((len(trange),))

    for mech in mechs:
        if mech != "Section":
            if datatype == "DataFrame":
                betas = base.xs(mech, level="mechanism").values.astype("float")
            else:
                betas = base[mech]
            beta_interp = interp1d(np.array(ts).astype(np.int_), betas)
            pfs[mech] = beta_to_pf(beta_interp(trange))
            # pfs[i] = ProbabilisticFunctions.beta_to_pf(betas)
            pnonfs = 1 - pfs[mech]
            if mech == "OVERFLOW":
                # pf_traject += np.max(pfs[i], axis=0)
                pf_traject = 1 - np.multiply(
                    1 - pf_traject, 1 - np.max(pfs[mech], axis=0)
                )
            else:
                # pf_traject += np.sum(pfs[i], axis=0)
                # pf_traject += 1-np.prod(pnonfs, axis=0)
                pf_traject = 1 - np.multiply(1 - pf_traject, np.prod(pnonfs, axis=0))

    ## INTERPOLATION AFTER COMBINATION:
    # pfail = interp1d(ts,pf_traject)
    # p_t1 = ProbabilisticFunctions.beta_to_pf(pfail(trange))
    # betafail = interp1d(ts, ProbabilisticFunctions.pf_to_beta(pf_traject),kind='linear')
    # beta_t = betafail(trange)
    # p_t = ProbabilisticFunctions.beta_to_pf(np.array(beta_t, dtype=np.float64))

    beta_t = pf_to_beta(pf_traject)
    p_t = pf_traject
    return beta_t, p_t
