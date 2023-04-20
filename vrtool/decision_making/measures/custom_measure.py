import copy

import numpy as np
import pandas as pd

from vrtool.decision_making.measures.measure_base import MeasureBase
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.flood_defence_system.dike_traject_info import DikeTrajectInfo


class CustomMeasure(MeasureBase):
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
        annual_dhc = (
            section.section_reliability.Mechanisms["Overflow"]
            .Reliability["0"]
            .Input.input["dhc(t)"]
        )
        if base_data["kruinhoogte_2075"].values > 0:
            self.parameters["h_crest_new"] = (
                base_data["kruinhoogte_2075"].values + 50 * annual_dhc
            )
        else:
            self.parameters["h_crest_new"] = None
        # TODO modify kruinhoogte_2075 to 2025 using change of crest in time.
        self.parameters["L_added"] = base_data["verlenging kwelweg"]
        self.measures["Cost"] = base_data["cost"].values[0]
        self.measures["Reliability"] = SectionReliability()
        self.measures["Reliability"].Mechanisms = {}

    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool = False,
    ):
        mechanism_names = list(dike_section.section_reliability.Mechanisms.keys())

        # first read and set the data:
        self.set_input(dike_section)

        # loop over mechanisms to modify the reliability
        for mechanism_name in mechanism_names:

            self.measures["Reliability"].Mechanisms[
                mechanism_name
            ] = MechanismReliabilityCollection(
                mechanism_name, "", self.config.T, self.config.t_0, 0
            )
            for ij in (
                self.measures["Reliability"]
                .Mechanisms[mechanism_name]
                .Reliability.keys()
            ):
                self.measures["Reliability"].Mechanisms[mechanism_name].Reliability[
                    ij
                ] = copy.deepcopy(
                    dike_section.section_reliability.Mechanisms[
                        mechanism_name
                    ].Reliability[ij]
                )

                # only adapt after year of implementation:
                if np.int_(ij) >= self.parameters["year"]:
                    # remove other input:
                    if mechanism_name == "Overflow":
                        if self.parameters["h_crest_new"] != None:
                            # type = simple
                            self.measures["Reliability"].Mechanisms[
                                mechanism_name
                            ].Reliability[ij].Input.input["h_crest"] = self.parameters[
                                "h_crest_new"
                            ]

                        # change crest
                    elif mechanism_name == "Piping":
                        self.measures["Reliability"].Mechanisms[
                            mechanism_name
                        ].Reliability[ij].Input.input["Lvoor"] += self.parameters[
                            "L_added"
                        ].values
                        # change Lvoor
                    else:
                        # Direct input: remove existing inputs and replace with beta
                        self.measures["Reliability"].Mechanisms[
                            mechanism_name
                        ].Reliability[ij].type = "DirectInput"
                        self.measures["Reliability"].Mechanisms[
                            mechanism_name
                        ].Reliability[ij].Input.input = {}
                        self.measures["Reliability"].Mechanisms[
                            mechanism_name
                        ].Reliability[ij].Input.input["beta"] = {}
                        for input in self.reliability_data[mechanism_name]:
                            # only read non-nan values:
                            if not np.isnan(
                                self.reliability_data[mechanism_name, input].values[0]
                            ):
                                self.measures["Reliability"].Mechanisms[
                                    mechanism_name
                                ].Reliability[ij].Input.input["beta"][
                                    input - self.t_0
                                ] = self.reliability_data[
                                    mechanism_name, input
                                ].values[
                                    0
                                ]
            self.measures["Reliability"].Mechanisms[mechanism_name].generateLCRProfile(
                dike_section.section_reliability.Load,
                mechanism=mechanism_name,
                trajectinfo=traject_info,
            )
        self.measures["Reliability"].calculate_section_reliability()
