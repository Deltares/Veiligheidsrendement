import copy
from pathlib import Path

import numpy as np

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.common_functions import (
    get_safety_factor_increase,
    sf_factor_piping,
)
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
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


class StabilityScreenMeasure(MeasureProtocol):
    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool = False,
    ) -> None:
        # To be added: year property to distinguish the same measure in year 2025 and 2045
        _lengths_stab_screen = [3.0, 6.0]
        self.measures = []
        for _length in _lengths_stab_screen:
            _modified_measure = {}
            _modified_measure["Stability Screen"] = "yes"
            _modified_measure["l_stab_screen"] = _length
            _depth = dike_section.cover_layer_thickness + _length
            _modified_measure["Cost"] = (
                self.unit_costs.sheetpile * _depth * dike_section.Length
            )
            _modified_measure["Reliability"] = self._get_configured_section_reliability(
                dike_section, traject_info, _length
            )
            _modified_measure["Reliability"].calculate_section_reliability(
                dike_section.get_cross_sectional_properties()
            )
            self.measures.append(_modified_measure)

    def _get_configured_section_reliability(
        self, dike_section: DikeSection, traject_info: DikeTrajectInfo, length: float
    ) -> SectionReliability:
        section_reliability = SectionReliability()

        mechanisms = (
            dike_section.section_reliability.failure_mechanisms.get_available_mechanisms()
        )
        for mechanism in mechanisms:
            calc_type = dike_section.mechanism_data[mechanism][0][1]
            mechanism_reliability_collection = (
                self._get_configured_mechanism_reliability_collection(
                    mechanism, calc_type, dike_section, traject_info, length
                )
            )
            section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
                mechanism_reliability_collection
            )

        return section_reliability

    def _get_configured_mechanism_reliability_collection(
        self,
        mechanism: MechanismEnum,
        calc_type: ComputationTypeEnum,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        length: float,
    ) -> MechanismReliabilityCollection:
        mechanism_reliability_collection = MechanismReliabilityCollection(
            mechanism, calc_type, self.config.T, self.config.t_0, 0
        )

        for (
            _year_to_calculate,
            _collection,
        ) in mechanism_reliability_collection.Reliability.items():
            _collection.Input = copy.deepcopy(
                dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                    mechanism
                )
                .Reliability[_year_to_calculate]
                .Input
            )

            dike_section_mechanism_reliability = dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                mechanism
            ).Reliability[
                _year_to_calculate
            ]
            if mechanism == MechanismEnum.STABILITY_INNER:
                self._configure_stability_inner(
                    _collection, _year_to_calculate, dike_section, length
                )
            elif mechanism == MechanismEnum.PIPING:
                self._copy_results(_collection, dike_section_mechanism_reliability)
                _collection.Input.input["sf_factor"] = sf_factor_piping(length)
            elif mechanism == MechanismEnum.OVERFLOW:
                self._copy_results(
                    _collection, dike_section_mechanism_reliability
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
        length: float,
    ) -> None:
        _calc_type = dike_section.mechanism_data[MechanismEnum.STABILITY_INNER][0][1]

        mechanism_reliability_input = mechanism_reliability.Input.input
        _safety_factor_increase = get_safety_factor_increase(length)
        _depth_screen = dike_section.cover_layer_thickness + length
        if _calc_type == ComputationTypeEnum.DSTABILITY:
            # Add screen to model
            _dstability_wrapper = DStabilityWrapper(
                Path(mechanism_reliability_input["STIXNAAM"]),
                Path(mechanism_reliability_input["DStability_exe_path"]),
            )
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
                    _original_name + "_stability_screen"
                ).name
            )

            if not _export_path.parent.exists():
                _export_path.parent.mkdir(parents=True)
            _dstability_wrapper.save_dstability_model(_export_path)
            _dstability_wrapper.rerun_stix()

            # Calculate reliability
            mechanism_reliability_input["beta"] = calculate_reliability(
                np.array([_dstability_wrapper.get_safety_factor()])
            )

        elif _calc_type == ComputationTypeEnum.SIMPLE:
            if int(year_to_calculate) >= self.parameters["year"]:
                mechanism_reliability_input["beta"] = calculate_reliability(
                    np.add(
                        calculate_safety_factor(mechanism_reliability_input["beta"]),
                        _safety_factor_increase,
                    )
                )
