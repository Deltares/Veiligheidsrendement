import copy

import numpy as np

from vrtool.decision_making.measures.common_functions import (
    determine_costs,
    probabilistic_design,
)
from vrtool.decision_making.measures.measure_base import MeasureBase
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.flood_defence_system.dike_traject_info import DikeTrajectInfo


class DiaphragmWallMeasure(MeasureBase):
    # type == 'Diaphragm Wall':
    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool = False,
    ):
        # To be added: year property to distinguish the same measure in year 2025 and 2045
        type = self.parameters["Type"]
        mechanism_names = dike_section.section_reliability.Mechanisms.keys()
        # StabilityInner and Piping reduced to 0, height is ok for overflow until 2125 (free of charge, also if there is a large height deficit).
        # It is assumed that the diaphragm wall is extendable after that.
        # Only 1 parameterized version with a lifetime of 100 years
        self.measures = {}
        self.measures["DiaphragmWall"] = "yes"
        self.measures["Cost"] = determine_costs(
            self.parameters, type, dike_section.Length, self.unit_costs
        )
        self.measures["Reliability"] = SectionReliability()
        self.measures["Reliability"].Mechanisms = {}
        for mechanism_name in mechanism_names:
            calc_type = dike_section.mechanism_data[mechanism_name][1]
            self.measures["Reliability"].Mechanisms[
                mechanism_name
            ] = MechanismReliabilityCollection(
                mechanism_name, calc_type, self.config.T, self.config.t_0, 0
            )
            for ij in (
                self.measures["Reliability"].Mechanisms[mechanism_name].Reliability.keys()
            ):
                self.measures["Reliability"].Mechanisms[mechanism_name].Reliability[
                    ij
                ].Input = copy.deepcopy(
                    dike_section.section_reliability.Mechanisms[mechanism_name]
                    .Reliability[ij]
                    .Input
                )
                if float(ij) >= self.parameters["year"]:
                    if mechanism_name == "Overflow":
                        Pt = traject_info.Pmax * traject_info.omegaOverflow
                        if (
                            dike_section.section_reliability.Mechanisms[mechanism_name]
                            .Reliability[ij]
                            .type
                            == "Simple"
                        ):
                            if hasattr(dike_section, "HBNRise_factor"):
                                hc = probabilistic_design(
                                    "h_crest",
                                    dike_section.section_reliability.Mechanisms[
                                        "Overflow"
                                    ]
                                    .Reliability[ij]
                                    .Input.input,
                                    p_t=Pt,
                                    t_0=self.t_0,
                                    horizon=self.parameters["year"] + 100,
                                    load_change=dike_section.HBNRise_factor
                                    * dike_section.YearlyWLRise,
                                    mechanism="Overflow",
                                )
                            else:
                                hc = probabilistic_design(
                                    "h_crest",
                                    dike_section.section_reliability.Mechanisms[
                                        "Overflow"
                                    ]
                                    .Reliability[ij]
                                    .Input.input,
                                    p_t=Pt,
                                    t_0=self.t_0,
                                    horizon=self.parameters["year"] + 100,
                                    load_change=None,
                                    mechanism="Overflow",
                                )
                        else:
                            hc = probabilistic_design(
                                "h_crest",
                                dike_section.section_reliability.Mechanisms["Overflow"]
                                .Reliability[ij]
                                .Input.input,
                                p_t=Pt,
                                t_0=self.t_0,
                                horizon=self.parameters["year"] + 100,
                                load_change=None,
                                type="HRING",
                                mechanism="Overflow",
                            )

                        self.measures["Reliability"].Mechanisms[mechanism_name].Reliability[
                            ij
                        ].Input.input["h_crest"] = np.max(
                            [
                                hc,
                                self.measures["Reliability"]
                                .Mechanisms[mechanism_name]
                                .Reliability[ij]
                                .Input.input["h_crest"],
                            ]
                        )  # should not become weaker!
                    elif mechanism_name == "StabilityInner" or mechanism_name == "Piping":
                        self.measures["Reliability"].Mechanisms[mechanism_name].Reliability[
                            ij
                        ].Input.input["Elimination"] = "yes"
                        self.measures["Reliability"].Mechanisms[mechanism_name].Reliability[
                            ij
                        ].Input.input["Pf_elim"] = self.parameters["P_solution"]
                        self.measures["Reliability"].Mechanisms[mechanism_name].Reliability[
                            ij
                        ].Input.input["Pf_with_elim"] = self.parameters["Pf_solution"]
            self.measures["Reliability"].Mechanisms[mechanism_name].generateLCRProfile(
                dike_section.section_reliability.Load,
                mechanism=mechanism_name,
                trajectinfo=traject_info,
            )
        self.measures["Reliability"].calculate_section_reliability()
