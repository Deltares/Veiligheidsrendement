import copy
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.common_functions import determine_costs
from vrtool.decision_making.measures.measure_base import MeasureBase
from vrtool.failure_mechanisms.stability_inner.dstability_wrapper import (
    DStabilityWrapper,
)
from vrtool.failure_mechanisms.stability_inner.stability_inner_functions import (
    calculate_reliability,
    calculate_safety_factor,
)
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class StabilityScreenMeasure(MeasureBase):
    # type == 'Stability Screen':
    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool = False,
        safety_factor_increase: float = 0.2,
    ):
        # To be added: year property to distinguish the same measure in year 2025 and 2045
        type = self.parameters["Type"]
        self.measures = {}
        self.measures["Stability Screen"] = "yes"
        self.parameters["Depth"] = self._get_depth(dike_section)
        self.measures["Cost"] = determine_costs(
            self.parameters, type, dike_section.Length, self.unit_costs
        )

        self.measures["Reliability"] = self._get_configured_section_reliability(
            dike_section, traject_info, safety_factor_increase
        )
        self.measures["Reliability"].calculate_section_reliability()

    def _get_depth(self, dike_section: DikeSection) -> float:
        """Gets the depth for the stability screen application.
        Args:
            dike_section (DikeSection): The section to retrieve the depth from.
        Raises:
            ValueError: Raised when there is no stability inner failure mechanism present.
        Returns:
            float: The depth to be used for the stability screen calculation.
        """
        stability_inner_reliability_collection = dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
            "StabilityInner"
        )
        if not stability_inner_reliability_collection:
            error_message = f'No StabilityInner present for stability screen measure at section "{dike_section.name}".'
            logging.error(error_message)
            raise ValueError(error_message)

        d_cover_input = stability_inner_reliability_collection.Reliability[
            "0"
        ].Input.input.get("d_cover", None)
        if d_cover_input:
            if d_cover_input.size > 1:
                logging.info("d_cover has more values than 1.")

            return max([d_cover_input[0] + 1.0, 8.0])
        else:
            # TODO remove shaky assumption on depth
            return 6.0

    def _get_configured_section_reliability(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        safety_factor_increase: float,
    ) -> SectionReliability:
        section_reliability = SectionReliability()

        mechanism_names = (
            dike_section.section_reliability.failure_mechanisms.get_available_mechanisms()
        )
        for mechanism_name in mechanism_names:
            calc_type = dike_section.mechanism_data[mechanism_name][1]
            mechanism_reliability_collection = (
                self._get_configured_mechanism_reliability_collection(
                    mechanism_name,
                    calc_type,
                    dike_section,
                    traject_info,
                    safety_factor_increase,
                )
            )
            section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
                mechanism_reliability_collection
            )

        return section_reliability

    def _get_configured_mechanism_reliability_collection(
        self,
        mechanism_name: str,
        calc_type: str,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        safety_factor_increase: float,
    ) -> MechanismReliabilityCollection:
        mechanism_reliability_collection = MechanismReliabilityCollection(
            mechanism_name, calc_type, self.config.T, self.config.t_0, 0
        )

        for year_to_calculate in mechanism_reliability_collection.Reliability.keys():
            mechanism_reliability_collection.Reliability[
                year_to_calculate
            ].Input = copy.deepcopy(
                dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                    mechanism_name
                )
                .Reliability[year_to_calculate]
                .Input
            )

            mechanism_reliability = mechanism_reliability_collection.Reliability[
                year_to_calculate
            ]
            dike_section_mechanism_reliability = dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                mechanism_name
            ).Reliability[
                year_to_calculate
            ]
            if float(year_to_calculate) >= self.parameters["year"]:
                if mechanism_name == "StabilityInner":
                    self._configure_stability_inner(
                        mechanism_reliability,
                        year_to_calculate,
                        dike_section,
                        safety_factor_increase,
                    )
                if mechanism_name in ["Piping", "Overflow"]:
                    self._copy_results(
                        mechanism_reliability, dike_section_mechanism_reliability
                    )  # No influence

        mechanism_reliability_collection.generate_LCR_profile(
            dike_section.section_reliability.load,
            traject_info=traject_info,
        )

        return mechanism_reliability_collection

    def _copy_results(
        self, target: MechanismReliability, source_input: MechanismReliability
    ) -> None:
        target.Input = copy.deepcopy(source_input.Input)

    def _configure_stability_inner(
        self,
        mechanism_reliability: MechanismReliability,
        year_to_calculate: str,
        dike_section: DikeSection,
        SFincrease: float = 0.2,
    ) -> None:

        _calc_type = dike_section.mechanism_data["StabilityInner"][1]

        mechanism_reliability_input = mechanism_reliability.Input.input
        if _calc_type == "DStability":

            # Add screen to model
            _dstability_wrapper = DStabilityWrapper(
                Path(mechanism_reliability_input["STIXNAAM"]),
                Path(mechanism_reliability_input["DStability_exe_path"]),
            )
            _depth_screen = self._get_depth(dike_section)
            _inner_toe = dike_section.InitialGeometry.loc["BIT"]
            _dstability_wrapper.add_stability_screen(
                bottom_screen=_inner_toe.z - _depth_screen, location=_inner_toe.x
            )

            # Save and run new model
            _original_name = _dstability_wrapper.stix_path.stem
            _export_path = (
                self.config.output_directory
                / "intermediate_result"
                / _dstability_wrapper.stix_path.with_stem(
                    _original_name + f"_stability_screen"
                ).name
            )

            if not _export_path.parent.exists():
                _export_path.parent.mkdir(parents=True)
            _dstability_wrapper.save_dstability_model(_export_path)
            _dstability_wrapper.rerun_stix()

            # Calculate reliaiblity
            mechanism_reliability_input["BETA"] = calculate_reliability(
                np.array(_dstability_wrapper.get_safety_factor())
            )

        elif _calc_type == "Simple":
            if int(year_to_calculate) >= self.parameters["year"]:
                if "SF_2025" in mechanism_reliability_input:
                    mechanism_reliability_input["SF_2025"] += SFincrease
                    mechanism_reliability_input["SF_2075"] += SFincrease
                elif "beta_2025" in mechanism_reliability.Input.input:
                    # convert to SF and back:
                    mechanism_reliability_input["beta_2025"] = calculate_reliability(
                        np.add(
                            calculate_safety_factor(
                                mechanism_reliability_input["beta_2025"]
                            ),
                            SFincrease,
                        )
                    )
                    mechanism_reliability_input["beta_2075"] = calculate_reliability(
                        np.add(
                            calculate_safety_factor(
                                mechanism_reliability_input["beta_2075"]
                            ),
                            SFincrease,
                        )
                    )
                else:
                    mechanism_reliability_input["BETA"] = calculate_reliability(
                        np.add(
                            calculate_safety_factor(
                                mechanism_reliability_input["BETA"]
                            ),
                            SFincrease,
                        )
                    )
