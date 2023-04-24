import copy
import logging

import numpy as np

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.common_functions import (
    determine_costs,
    determine_new_geometry,
    implement_berm_widening,
)
from vrtool.decision_making.measures.measure_base import MeasureBase
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.decision_making.measures.modified_dike_geometry_measure_input import (
    ModifiedDikeGeometryMeasureInput,
)


class SoilReinforcementMeasure(MeasureBase):
    # type == 'Soil reinforcement':
    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        plot_dir: bool = False,
        preserve_slope: bool = False,
    ):
        # def evaluateMeasure(self, DikeSection, TrajectInfo, preserve_slope=False):
        # To be added: year property to distinguish the same measure in year 2025 and 2045
        # Measure.__init__(self,inputs)
        # self. parameters = measure.parameters
        type = self.parameters["Type"]
        if self.parameters["StabilityScreen"] == "yes":
            self.parameters["Depth"] = self._get_depth(dike_section)

        modified_dike_geometry_measures = self._get_modified_dike_geometry_measures(
            dike_section, preserve_slope, plot_dir
        )

        self.measures = []
        for modified_dike_geometry_measure in modified_dike_geometry_measures:
            self.measures.append({})
            self.measures[-1]["dcrest"] = modified_dike_geometry_measure.d_crest
            self.measures[-1]["dberm"] = modified_dike_geometry_measure.d_berm
            self.measures[-1]["Cost"] = determine_costs(
                self.parameters,
                type,
                dike_section.Length,
                self.unit_costs,
                dcrest=modified_dike_geometry_measure.d_crest,
                dberm_in=int(modified_dike_geometry_measure.d_house),
                housing=dike_section.houses,
                area_extra=modified_dike_geometry_measure.area_extra,
                area_excavated=modified_dike_geometry_measure.area_excavated,
                direction=self.parameters["Direction"],
                section=dike_section.name,
            )

            self.measures[-1]["Reliability"] = self._get_configured_section_reliability(
                dike_section, traject_info
            )
            self.measures[-1]["Reliability"].calculate_section_reliability()

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
        d_cover_input = (
            dike_section.section_reliability.Mechanisms["StabilityInner"]
            .Reliability["0"]
            .Input.input.get("d_cover", None)
        )

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

            inputs.append(
                ModifiedDikeGeometryMeasureInput.from_dictionary(
                    measure_input_dictionary
                )
            )

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
        else:
            return determine_new_geometry(
                dike_modification,
                self.parameters["Direction"],
                self.parameters["max_outward"],
                copy.deepcopy(dike_section.InitialGeometry),
                self.geometry_plot,
                **{"plot_dir": plot_dir, "slope_in": slope_in},
            )

    def _get_configured_section_reliability(
        self, dike_section: DikeSection, traject_info: DikeTrajectInfo
    ) -> SectionReliability:
        section_reliability = SectionReliability()
        section_reliability.Mechanisms = {}

        mechanism_names = dike_section.section_reliability.Mechanisms.keys()
        for mechanism_name in mechanism_names:
            calc_type = dike_section.mechanism_data[mechanism_name][1]
            section_reliability.Mechanisms[
                mechanism_name
            ] = self._get_configured_mechanism_reliability_collection(
                mechanism_name, calc_type, dike_section, traject_info
            )

        return section_reliability

    def _get_configured_mechanism_reliability_collection(
        self,
        mechanism_name: str,
        calc_type: str,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
    ) -> MechanismReliabilityCollection:
        mechanism_reliability_collection = MechanismReliabilityCollection(
            mechanism_name,
            calc_type,
            self.config.T,
            self.config.t_0,
            self.parameters["year"],
        )
        for (
            year_to_calculate,
            reliability_input,
        ) in mechanism_reliability_collection.Reliability.items():
            # for all time steps considered.
            # first copy the data
            reliability_input = copy.deepcopy(
                dike_section.section_reliability.Mechanisms[mechanism_name]
                .Reliability[year_to_calculate]
                .Input
            )
            # Adapt inputs for reliability calculation, but only after year of implementation.
            if float(year_to_calculate) >= self.parameters["year"]:
                reliability_input.input = implement_berm_widening(
                    input=reliability_input.input,
                    measure_input=self.measures[-1],
                    measure_parameters=self.parameters,
                    mechanism=mechanism_name,
                    computation_type=calc_type,
                )
            # put them back in the object
            mechanism_reliability_collection.Reliability[
                year_to_calculate
            ].Input = reliability_input

        mechanism_reliability_collection.generateLCRProfile(
            dike_section.section_reliability.Load,
            mechanism=mechanism_name,
            trajectinfo=traject_info,
        )

        return mechanism_reliability_collection
