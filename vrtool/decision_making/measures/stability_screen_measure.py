import copy

import numpy as np

from vrtool.decision_making.measures.common_functions import determine_costs
from vrtool.decision_making.measures.measure_base import MeasureBase
from vrtool.failure_mechanisms.stability_inner.stability_inner_simple import (
    StabilityInnerSimple,
)
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class StabilityScreenMeasure(MeasureBase):
    # type == 'Stability Screen':
    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: dict[str, any],
        preserve_slope: bool = False,
        SFincrease: float = 0.2,
    ):
        # To be added: year property to distinguish the same measure in year 2025 and 2045
        type = self.parameters["Type"]
        mechanisms = dike_section.section_reliability.Mechanisms.keys()
        self.measures = {}
        self.measures["Stability Screen"] = "yes"
        if (
            "d_cover"
            in dike_section.section_reliability.Mechanisms["StabilityInner"]
            .Reliability["0"]
            .Input.input
        ):
            self.parameters["Depth"] = max(
                [
                    dike_section.section_reliability.Mechanisms["StabilityInner"]
                    .Reliability["0"]
                    .Input.input["d_cover"][0]
                    + 1.0,
                    8.0,
                ]
            )
        else:
            # TODO remove shaky assumption on depth
            self.parameters["Depth"] = 6.0
        self.measures["Cost"] = determine_costs(
            self.parameters, type, dike_section.Length, self.unit_costs
        )
        self.measures["Reliability"] = SectionReliability()
        self.measures["Reliability"].Mechanisms = {}
        for i in mechanisms:
            calc_type = dike_section.mechanism_data[i][1]
            self.measures["Reliability"].Mechanisms[i] = MechanismReliabilityCollection(
                i, calc_type, self.config
            )
            for ij in self.measures["Reliability"].Mechanisms[i].Reliability.keys():
                self.measures["Reliability"].Mechanisms[i].Reliability[
                    ij
                ].Input = copy.deepcopy(
                    dike_section.section_reliability.Mechanisms[i].Reliability[ij].Input
                )
                if i == "Overflow" or i == "Piping":  # Copy results
                    self.measures["Reliability"].Mechanisms[i].Reliability[
                        ij
                    ] = copy.deepcopy(
                        dike_section.section_reliability.Mechanisms[i].Reliability[ij]
                    )
                    pass  # no influence
                elif i == "StabilityInner":
                    self.measures["Reliability"].Mechanisms[i].Reliability[
                        ij
                    ].Input = copy.deepcopy(
                        dike_section.section_reliability.Mechanisms[i]
                        .Reliability[ij]
                        .Input
                    )
                    if int(ij) >= self.parameters["year"]:
                        if (
                            "SF_2025"
                            in self.measures["Reliability"]
                            .Mechanisms[i]
                            .Reliability[ij]
                            .Input.input
                        ):
                            self.measures["Reliability"].Mechanisms[i].Reliability[
                                ij
                            ].Input.input["SF_2025"] += SFincrease
                            self.measures["Reliability"].Mechanisms[i].Reliability[
                                ij
                            ].Input.input["SF_2075"] += SFincrease
                        elif (
                            "beta_2025"
                            in self.measures["Reliability"]
                            .Mechanisms[i]
                            .Reliability[ij]
                            .Input.input
                        ):
                            # convert to SF and back:
                            self.measures["Reliability"].Mechanisms[i].Reliability[
                                ij
                            ].Input.input[
                                "beta_2025"
                            ] = StabilityInnerSimple.calculate_reliability(
                                np.add(
                                    StabilityInnerSimple.calculate_safety_factor(
                                        self.measures["Reliability"]
                                        .Mechanisms[i]
                                        .Reliability[ij]
                                        .Input.input["beta_2025"]
                                    ),
                                    SFincrease,
                                )
                            )
                            self.measures["Reliability"].Mechanisms[i].Reliability[
                                ij
                            ].Input.input[
                                "beta_2075"
                            ] = StabilityInnerSimple.calculate_reliability(
                                np.add(
                                    StabilityInnerSimple.calculate_safety_factor(
                                        self.measures["Reliability"]
                                        .Mechanisms[i]
                                        .Reliability[ij]
                                        .Input.input["beta_2075"]
                                    ),
                                    SFincrease,
                                )
                            )
                        else:
                            self.measures["Reliability"].Mechanisms[i].Reliability[
                                ij
                            ].Input.input["SF"] = np.add(
                                self.measures["Reliability"]
                                .Mechanisms[i]
                                .Reliability[ij]
                                .Input.input["SF"],
                                SFincrease,
                            )
                            self.measures["Reliability"].Mechanisms[i].Reliability[
                                ij
                            ].Input.input[
                                "BETA"
                            ] = StabilityInnerSimple.calculate_reliability(
                                self.measures["Reliability"]
                                .Mechanisms[i]
                                .Reliability[ij]
                                .Input.input["SF"]
                            )

            self.measures["Reliability"].Mechanisms[i].generateLCRProfile(
                dike_section.section_reliability.Load,
                mechanism=i,
                trajectinfo=traject_info,
            )
        self.measures["Reliability"].calculate_section_reliability()
