import copy

import numpy as np

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.common_functions import determine_costs
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class VerticalGeotextileMeasure(MeasureProtocol):
    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool = False,
    ):
        # To be added: year property to distinguish the same measure in year 2025 and 2045
        type = self.parameters["Type"]

        # No influence on overflow and stability
        # Only 1 parameterized version with a lifetime of 50 years
        self.measures = {}
        self.measures["VZG"] = "yes"
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
            dike_section_mechanism_reliability = dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                mechanism
            ).Reliability[
                year_to_calculate
            ]
            if mechanism == MechanismEnum.PIPING:
                self._configure_piping(
                    mechanism_reliability,
                    year_to_calculate,
                    dike_section_mechanism_reliability,
                )
            if mechanism in [MechanismEnum.OVERFLOW, MechanismEnum.STABILITY_INNER]:
                self._copy_results(
                    mechanism_reliability, dike_section_mechanism_reliability
                )

        mechanism_reliability_collection.generate_LCR_profile(
            dike_section.section_reliability.load,
            traject_info=traject_info,
        )

        return mechanism_reliability_collection

    def _copy_results(
        self, target: MechanismReliability, source_input: MechanismReliability
    ) -> None:
        target.Input = copy.deepcopy(source_input.Input)

    def _configure_piping(
        self,
        mechanism_reliability: MechanismReliability,
        year_to_calculate: str,
        dike_section_piping_reliability: MechanismReliability,
    ) -> None:
        if int(year_to_calculate) < self.parameters["year"]:
            self._copy_results(mechanism_reliability, dike_section_piping_reliability)

        mechanism_reliability.Input.input["elimination"] = "yes"
        mechanism_reliability.Input.input["pf_elim"] = self.parameters["P_solution"]
        mechanism_reliability.Input.input["pf_with_elim"] = np.min(
            [self.parameters["Pf_solution"], 1.0e-16]
        )
