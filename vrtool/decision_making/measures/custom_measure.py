import copy
import logging
from typing import Optional

import numpy as np
import pandas as pd

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.measure_base import MeasureProtocol
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class CustomMeasure(MeasureProtocol):
    def set_input(self, section: DikeSection):
        try:
            try:
                data = pd.read_csv(
                    self.input_directory.joinpath("Measures", self.parameters["File"])
                )
            except:
                data = pd.read_csv(self.parameters["File"])
            reliability_headers = []
            for i, element in enumerate(list(data.columns)):
                # find and split headers
                if "beta" in element:
                    reliability_headers.append(element.split("_"))
                    if "start_id" not in locals():
                        start_id = i
            # make 2 dataframes: 1 with base data and 1 with reliability data
            base_data = data.iloc[:, 0:start_id]
            reliability_data = data.iloc[:, start_id:]
            reliability_data.columns = pd.MultiIndex.from_arrays(
                [
                    np.array(reliability_headers)[:, 1],
                    np.array(reliability_headers)[:, 2].astype(np.int32),
                ],
                names=["mechanism", "year"],
            )
            # TODO reindex the reliability data such that the mechanism is the index and year the column. Now it is a multiindex, hwich works as well but is not as nice.
        except:
            raise Exception(self.parameters["File"] + " not found.")
        # self.base_data = base_data
        self.reliability_data = reliability_data
        self.measures = {}
        self.parameters["year"] = np.int32(base_data["year"] - self.t_0)

        # TODO check these values:
        # base_data['kruinhoogte']=6.
        # base_data['extra kwelweg'] = 10.
        # TODO modify kruinhoogte_2075 to 2025 using change of crest in time.
        self.parameters["h_crest_new"] = self._get_h_crest_new(section, base_data)
        self.parameters["L_added"] = base_data["verlenging kwelweg"]
        self.measures["Cost"] = base_data["cost"].values[0]

    def _get_h_crest_new(
        self, section: DikeSection, base_data: pd.DataFrame
    ) -> Optional[float]:
        overflow_reliability_collection = section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
            "Overflow"
        )
        if not overflow_reliability_collection:
            logging.warn(f'Overflow data is not present in section "{section.name}"')

        else:
            if base_data["kruinhoogte_2075"].values > 0:
                annual_dhc = overflow_reliability_collection.Reliability[
                    "0"
                ].Input.input["dhc(t)"]
                return base_data["kruinhoogte_2075"].values + 50 * annual_dhc
            else:
                logging.warn("kruinhoogte 2075 is not found.")

        return None

    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool = False,
    ):
        # first read and set the data:
        self.set_input(dike_section)

        self.measures["Reliability"] = self._get_configured_section_reliability(
            dike_section, traject_info
        )
        self.measures["Reliability"].calculate_section_reliability()

    def _get_configured_section_reliability(
        self, dike_section: DikeSection, traject_info: DikeTrajectInfo
    ) -> SectionReliability:
        section_reliability = SectionReliability()

        mechanism_names = (
            dike_section.section_reliability.failure_mechanisms.get_available_mechanisms()
        )
        for mechanism_name in mechanism_names:
            mechanism_reliability_collection = (
                self._get_configured_mechanism_reliability_collection(
                    mechanism_name, dike_section, traject_info
                )
            )
            section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
                mechanism_reliability_collection
            )

        return section_reliability

    def _get_configured_mechanism_reliability_collection(
        self,
        mechanism_name: str,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
    ) -> MechanismReliabilityCollection:
        mechanism_reliability_collection = MechanismReliabilityCollection(
            mechanism_name, "", self.config.T, self.config.t_0, 0
        )

        for year_to_calculate in mechanism_reliability_collection.Reliability.keys():
            mechanism_reliability_collection.Reliability[
                year_to_calculate
            ] = copy.deepcopy(
                dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                    mechanism_name
                ).Reliability[
                    year_to_calculate
                ]
            )

            mechanism_reliability = mechanism_reliability_collection.Reliability[
                year_to_calculate
            ]
            if np.int_(year_to_calculate) >= self.parameters["year"]:
                if mechanism_name == "Overflow":
                    self._configure_overflow(mechanism_reliability)
                elif mechanism_name == "Piping":
                    self._configure_piping(mechanism_reliability)
                else:
                    self._configure_other(mechanism_reliability, mechanism_name)

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
        self, mechanism_reliability: MechanismReliability, mechanism_name: str
    ) -> None:
        # Direct input: remove existing inputs and replace with beta
        mechanism_reliability.type = "DirectInput"
        mechanism_reliability.Input.input = {}
        mechanism_reliability.Input.input["beta"] = {}

        for input in self.reliability_data[mechanism_name]:
            # only read non-nan values:
            if not np.isnan(self.reliability_data[mechanism_name, input].values[0]):
                mechanism_reliability.Input.input["beta"][
                    input - self.t_0
                ] = self.reliability_data[mechanism_name, input].values[0]
