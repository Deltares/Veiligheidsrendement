import copy
import logging
from pathlib import Path

import numpy as np

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.berm_widening_dstability import BermWideningDStability
from vrtool.decision_making.measures.common_functions import determine_costs, determine_new_geometry
from vrtool.decision_making.measures.measure_base import MeasureBase
from vrtool.decision_making.measures.modified_dike_geometry_measure_input import ModifiedDikeGeometryMeasureInput
from vrtool.failure_mechanisms.stability_inner.dstability_wrapper import DStabilityWrapper
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
        self.parameters["Depth"] = self._get_depth(dike_section)

        def get_measure_data(
                modified_measure: ModifiedDikeGeometryMeasureInput,
        ) -> dict:
            _modified_measure = {}
            _modified_measure["Stability Screen"] = "yes"
            _modified_measure["geometry"] = modified_measure.modified_geometry
            _modified_measure["dcrest"] = modified_measure.d_crest
            _modified_measure["dberm"] = modified_measure.d_berm
            _modified_measure["Cost"] = determine_costs(
                self.parameters, type, dike_section.Length, self.unit_costs
            )
            _modified_measure["Reliability"] = self._get_configured_section_reliability(
                dike_section, traject_info, safety_factor_increase, _modified_measure
            )
            _modified_measure["Reliability"].calculate_section_reliability()
            return _modified_measure

        modified_dike_geometry_measures = self._get_modified_dike_geometry_measures(
            dike_section, preserve_slope
        )

        self.measures = list(map(get_measure_data, modified_dike_geometry_measures))

    def _get_crest_range(self) -> np.ndarray:
        crest_step = self.crest_step
        return np.linspace(
            self.parameters["dcrest_min"],
            self.parameters["dcrest_max"],
            np.int_(
                1
                + (self.parameters["dcrest_max"] - self.parameters["dcrest_min"])
                / crest_step
            ),
        )

    def _get_berm_range(self) -> np.ndarray:
        """Generates the range of the berm.

        Raises:
            Exception: Raised when an unknown direction is specified

        Returns:
            np.ndarray: A collection of the berm range.
        """
        berm_step = self.berm_step
        # TODO: CLEAN UP, make distinction between inwards and outwards, so xin, xout and y,and adapt DetermineNewGeometry
        if self.parameters["Direction"] == "outward":
            if np.size(berm_step) > 1:
                max_berm = (
                        self.parameters["max_outward"] + self.parameters["max_inward"]
                )
                return berm_step[: len(np.where((berm_step <= max_berm))[0])]
            else:
                return np.linspace(
                    0.0,
                    self.parameters["max_outward"],
                    np.int_(1 + (self.parameters["max_outward"] / berm_step)),
                )
        elif self.parameters["Direction"] == "inward":
            if np.size(berm_step) > 1:
                max_berm = self.parameters["max_inward"]
                return berm_step[: len(np.where((berm_step <= max_berm))[0])]
            else:
                return np.linspace(
                    0.0,
                    self.parameters["max_inward"],
                    np.int_(1 + (self.parameters["max_inward"] / berm_step)),
                )
        else:
            raise Exception("unkown direction")

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

    def _get_modified_dike_geometry_measures(
            self,
            dike_section: DikeSection,
            preserve_slope: bool,
            plot_dir: bool = False,
    ) -> list[ModifiedDikeGeometryMeasureInput]:
        crest_range = self._get_crest_range()
        berm_range = self._get_berm_range()

        dike_modifications = [
            (modified_crest, modified_berm)
            for modified_crest in crest_range
            for modified_berm in berm_range
        ]

        inputs = []
        for dike_modification in dike_modifications:
            modified_geometry_properties = self._determine_new_geometry(
                dike_section, dike_modification, preserve_slope, plot_dir
            )

            measure_input_dictionary = {
                "d_crest": dike_modification[0],
                "d_berm": dike_modification[1],
                "modified_geometry": modified_geometry_properties[0],
                "area_extra": modified_geometry_properties[1],
                "area_excavated": modified_geometry_properties[2],
                "d_house": modified_geometry_properties[3],
            }

            inputs.append(ModifiedDikeGeometryMeasureInput(**measure_input_dictionary))

        return inputs

    def _determine_new_geometry(
            self,
            dike_section: DikeSection,
            dike_modification: tuple[float],
            preserve_slope: bool,
            plot_dir: bool,
    ) -> list:
        if not preserve_slope:
            slope_in = 4
            slope_out = 3  # inner and outer slope
        else:
            slope_in = False
            slope_out = False

        if hasattr(dike_section, "Kruinhoogte"):
            if dike_section.Kruinhoogte != np.max(dike_section.InitialGeometry.z):
                # In case the crest is unequal to the Kruinhoogte, that value should be given as input as well
                return determine_new_geometry(
                    dike_modification,
                    self.parameters["Direction"],
                    self.parameters["max_outward"],
                    copy.deepcopy(dike_section.InitialGeometry),
                    self.geometry_plot,
                    **{
                        "plot_dir": plot_dir,
                        "slope_in": slope_in,
                        "crest_extra": dike_section.Kruinhoogte,
                    },
                )
            else:
                return determine_new_geometry(
                    dike_modification,
                    self.parameters["Direction"],
                    self.parameters["max_outward"],
                    copy.deepcopy(dike_section.InitialGeometry),
                    self.geometry_plot,
                    **{"plot_dir": plot_dir, "slope_in": slope_in},
                )

        return determine_new_geometry(
            dike_modification,
            self.parameters["Direction"],
            self.parameters["max_outward"],
            copy.deepcopy(dike_section.InitialGeometry),
            self.geometry_plot,
            **{"plot_dir": plot_dir, "slope_in": slope_in},
        )

    def _get_configured_section_reliability(
            self,
            dike_section: DikeSection,
            traject_info: DikeTrajectInfo,
            safety_factor_increase: float,
            modified_geometry_measure: dict,

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
                    modified_geometry_measure
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
            modified_geometry_measure: dict,

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
            dike_section_mechanism_reliability = \
                dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                    mechanism_name
                ).Reliability[
                    year_to_calculate
                ]
            if float(year_to_calculate) >= self.parameters["year"]:
                if mechanism_name == "StabilityInner":
                    if calc_type == "DStability":
                        self._configure_stability_inner_dstability(mechanism_reliability,
                                                                   dike_section,
                                                                   modified_geometry_measure,
                                                                   self.config.output_directory / "intermediate_result")
                    else:
                        self._configure_stability_inner(
                            mechanism_reliability, year_to_calculate, safety_factor_increase
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
            SFincrease: float = 0.2,
    ) -> None:

        mechanism_reliability_input = mechanism_reliability.Input.input
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
                        calculate_safety_factor(mechanism_reliability_input["BETA"]),
                        SFincrease,
                    )
                )

    def _configure_stability_inner_dstability(self, mechanism_reliability: MechanismReliability,
                                              dike_section: DikeSection, modified_geometry_measure: dict,
                                              path_intermediate_stix: Path) -> None:
        """
        Call the DStability wrapper to add the stability screen and the berm widening to the DStability model. The
        screen is always placed at the BIT.

        Args:
            mechanism_reliability: mechanism input for the StabilityInner mechanism
            dike_section: dike section
            modified_geometry_measure: geometry input of the measure
            path_intermediate_stix: Path to the directory where the intermediate stix will be saved

        Return:
            None
        """
        _mechanism_reliability_input = mechanism_reliability.Input.input
        _depth = self._get_depth(dike_section)
        _BIT = modified_geometry_measure['geometry'].loc["BIT"]

        _dstability_wrapper = DStabilityWrapper(stix_path=Path(_mechanism_reliability_input['STIXNAAM']),
                                                externals_path=Path(_mechanism_reliability_input['DStability_exe_path']))

        # 1. Add the stability screen to the DStability model
        _dstability_wrapper.add_stability_screen(bottom_screen=_BIT.z - _depth, location=_BIT.x)

        # 2. Modify the geometry of the Berm in the same way as for SoilReinforcementMeasure
        _dstability_berm_widening = BermWideningDStability(
            measure_input=modified_geometry_measure, dstability_wrapper=_dstability_wrapper
        )

        #  Update the name of the stix file in the mechanism input dictionary, this is the stix that will be used
        # by the calculator later on. In this case, we need to force the wrapper to recalculate the DStability
        # model, hence RERUN_STIX set to True.
        _mechanism_reliability_input[
            "STIXNAAM"
        ] = _dstability_berm_widening.create_new_dstability_model(
            path_intermediate_stix
        )
        _mechanism_reliability_input["RERUN_STIX"] = True
