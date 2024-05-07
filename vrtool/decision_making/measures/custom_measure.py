import copy
import logging
from typing import Optional

import numpy as np
import pandas as pd

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class CustomMeasure(MeasureProtocol):
    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool = False,
    ):
        self.measures["Reliability"] = self._get_configured_section_reliability(
            dike_section, traject_info
        )
        self.measures["Reliability"].calculate_section_reliability()

    def _get_configured_section_reliability(
        self, dike_section: DikeSection, traject_info: DikeTrajectInfo
    ) -> SectionReliability:
        section_reliability = SectionReliability()

        mechanisms = (
            dike_section.section_reliability.failure_mechanisms.get_available_mechanisms()
        )
        for mechanism in mechanisms:
            mechanism_reliability_collection = (
                self._get_configured_mechanism_reliability_collection(
                    mechanism, dike_section, traject_info
                )
            )
            section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
                mechanism_reliability_collection
            )

        return section_reliability

    def _get_configured_mechanism_reliability_collection(
        self,
        mechanism: MechanismEnum,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
    ) -> MechanismReliabilityCollection:
        mechanism_reliability_collection = MechanismReliabilityCollection(
            mechanism, "", self.config.T, self.config.t_0, 0
        )

        for year_to_calculate in mechanism_reliability_collection.Reliability.keys():
            mechanism_reliability_collection.Reliability[
                year_to_calculate
            ] = copy.deepcopy(
                dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                    mechanism
                ).Reliability[
                    year_to_calculate
                ]
            )

            mechanism_reliability = mechanism_reliability_collection.Reliability[
                year_to_calculate
            ]
            if np.int_(year_to_calculate) >= self.parameters["year"]:
                if mechanism == MechanismEnum.OVERFLOW:
                    self._configure_overflow(mechanism_reliability)
                elif mechanism == MechanismEnum.PIPING:
                    self._configure_piping(mechanism_reliability)
                else:
                    self._configure_other(mechanism_reliability, mechanism)

        mechanism_reliability_collection.generate_LCR_profile(
            dike_section.section_reliability.load,
            traject_info=traject_info,
        )

        return mechanism_reliability_collection

    def _configure_overflow(self, mechanism_reliability: MechanismReliability) -> None:
        if self.parameters["h_crest_new"] != None:
            # type = simple
            mechanism_reliability.Input.input["h_crest"] = self.parameters[
                "h_crest_new"
            ]

        # change crest

    def _configure_piping(self, mechanism_reliability: MechanismReliability) -> None:
        mechanism_reliability.Input.input["Lvoor"] += self.parameters["L_added"].values
        # change Lvoor

    def _configure_other(
        self, mechanism_reliability: MechanismReliability, mechanism: MechanismEnum
    ) -> None:
        # Direct input: remove existing inputs and replace with beta
        mechanism_reliability.mechanism_type = "DirectInput"
        mechanism_reliability.Input.input = {}
        mechanism_reliability.Input.input["beta"] = {}

        for _reliability_input in self.reliability_data[mechanism.name]:
            # only read non-nan values:
            if not np.isnan(
                self.reliability_data[mechanism.name, _reliability_input].values[0]
            ):
                mechanism_reliability.Input.input["beta"][
                    _reliability_input - self.t_0
                ] = self.reliability_data[mechanism.name, _reliability_input].values[0]
