import copy

import numpy as np

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums import MechanismEnum
from vrtool.decision_making.measures.common_functions import (
    determine_costs,
    probabilistic_design,
)
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class DiaphragmWallMeasure(MeasureProtocol):
    # type == 'Diaphragm Wall':
    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool = False,
    ):
        # To be added: year property to distinguish the same measure in year 2025 and 2045
        type = self.parameters["Type"]
        # StabilityInner and Piping reduced to 0, height is ok for overflow until 2125 (free of charge, also if there is a large height deficit).
        # It is assumed that the diaphragm wall is extendable after that.
        # Only 1 parameterized version with a lifetime of 100 years
        self.measures = {}
        self.measures["DiaphragmWall"] = "yes"
        self.measures["Cost"] = determine_costs(
            self.parameters,
            type,
            dike_section.Length,
            self.parameters.get("Depth", float("nan")),
            self.unit_costs,
        )
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
            calc_type = dike_section.mechanism_data[mechanism][0][1]
            mechanism_reliability_collection = (
                self._get_configured_mechanism_reliability_collection(
                    mechanism, calc_type, dike_section, traject_info
                )
            )
            section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
                mechanism_reliability_collection
            )

        return section_reliability

    def _get_configured_mechanism_reliability_collection(
        self,
        mechanism: MechanismEnum,
        calc_type: str,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
    ) -> MechanismReliabilityCollection:
        mechanism_reliability_collection = MechanismReliabilityCollection(
            mechanism, calc_type, self.config.T, self.config.t_0, 0
        )

        for year_to_calculate in mechanism_reliability_collection.Reliability.keys():
            mechanism_reliability_collection.Reliability[
                year_to_calculate
            ].Input = copy.deepcopy(
                dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                    mechanism
                )
                .Reliability[year_to_calculate]
                .Input
            )

            mechanism_reliability = mechanism_reliability_collection.Reliability[
                year_to_calculate
            ]
            if float(year_to_calculate) >= self.parameters["year"]:
                if mechanism.name == "OVERFLOW":
                    self._configure_overflow(
                        mechanism_reliability, traject_info, dike_section
                    )
                if mechanism.name in ["PIPING", "STABILITY_INNER"]:
                    self._configure_piping_or_stability_inner(mechanism_reliability)

        mechanism_reliability_collection.generate_LCR_profile(
            dike_section.section_reliability.load,
            traject_info=traject_info,
        )

        return mechanism_reliability_collection

    def _configure_overflow(
        self,
        mechanism_reliability: MechanismReliability,
        traject_info: DikeTrajectInfo,
        dike_section: DikeSection,
    ) -> None:
        Pt = traject_info.Pmax * traject_info.omegaOverflow

        mechanism_input = mechanism_reliability.Input.input
        if mechanism_reliability.mechanism_type == "Simple":
            if hasattr(dike_section, "HBNRise_factor"):
                hc = probabilistic_design(
                    "h_crest",
                    mechanism_input,
                    p_t=Pt,
                    t_0=self.t_0,
                    horizon=self.parameters["year"] + 100,
                    load_change=dike_section.HBNRise_factor * dike_section.YearlyWLRise,
                    mechanism=MechanismEnum["OVERFLOW"],
                )
            else:
                hc = probabilistic_design(
                    "h_crest",
                    mechanism_input,
                    p_t=Pt,
                    t_0=self.t_0,
                    horizon=self.parameters["year"] + 100,
                    load_change=None,
                    mechanism=MechanismEnum["OVERFLOW"],
                )
        else:
            hc = probabilistic_design(
                "h_crest",
                mechanism_input,
                p_t=Pt,
                t_0=self.t_0,
                horizon=self.parameters["year"] + 100,
                load_change=None,
                type="HRING",
                mechanism=MechanismEnum["OVERFLOW"],
            )

        mechanism_input["h_crest"] = np.max(
            [hc, mechanism_input["h_crest"]]
        )  # should not become weaker!

    def _configure_piping_or_stability_inner(
        self, mechanism_reliability: MechanismReliability
    ) -> None:
        mechanism_reliability.Input.input["elimination"] = "yes"
        mechanism_reliability.Input.input["pf_elim"] = self.parameters["P_solution"]
        mechanism_reliability.Input.input["pf_with_elim"] = self.parameters[
            "Pf_solution"
        ]
