import copy

import numpy as np

from vrtool.decision_making.measures.common_functions import determine_costs
from vrtool.decision_making.measures.measure_base import MeasureBase
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.flood_defence_system.dike_traject_info import DikeTrajectInfo


class VerticalGeotextileMeasure(MeasureBase):
    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool = False,
    ):
        # To be added: year property to distinguish the same measure in year 2025 and 2045
        type = self.parameters["Type"]
        mechanisms = dike_section.section_reliability.Mechanisms.keys()

        # No influence on overflow and stability
        # Only 1 parameterized version with a lifetime of 50 years
        self.measures = {}
        self.measures["VZG"] = "yes"
        self.measures["Cost"] = determine_costs(
            self.parameters, type, dike_section.Length, self.unit_costs
        )
        self.measures["Reliability"] = SectionReliability()
        self.measures["Reliability"].Mechanisms = {}

        for mechanism in mechanisms:
            calc_type = dike_section.mechanism_data[mechanism][1]
            self.measures["Reliability"].Mechanisms[
                mechanism
            ] = MechanismReliabilityCollection(
                mechanism, calc_type, self.config.T, self.config.t_0
            )
            for ij in (
                self.measures["Reliability"].Mechanisms[mechanism].Reliability.keys()
            ):
                self.measures["Reliability"].Mechanisms[mechanism].Reliability[
                    ij
                ].Input = copy.deepcopy(
                    dike_section.section_reliability.Mechanisms[mechanism]
                    .Reliability[ij]
                    .Input
                )
                if (
                    mechanism == "Overflow"
                    or mechanism == "StabilityInner"
                    or (mechanism == "Piping" and int(ij) < self.parameters["year"])
                ):  # Copy results
                    self.measures["Reliability"].Mechanisms[mechanism].Reliability[
                        ij
                    ] = copy.deepcopy(
                        dike_section.section_reliability.Mechanisms[
                            mechanism
                        ].Reliability[ij]
                    )
                elif mechanism == "Piping" and int(ij) >= self.parameters["year"]:
                    self.measures["Reliability"].Mechanisms[mechanism].Reliability[
                        ij
                    ].Input.input["Elimination"] = "yes"
                    self.measures["Reliability"].Mechanisms[mechanism].Reliability[
                        ij
                    ].Input.input["Pf_elim"] = self.parameters["P_solution"]
                    self.measures["Reliability"].Mechanisms[mechanism].Reliability[
                        ij
                    ].Input.input["Pf_with_elim"] = np.min(
                        [self.parameters["Pf_solution"], 1.0e-16]
                    )
            self.measures["Reliability"].Mechanisms[mechanism].generateLCRProfile(
                dike_section.section_reliability.Load,
                mechanism=mechanism,
                trajectinfo=traject_info,
            )
        self.measures["Reliability"].calculate_section_reliability()
