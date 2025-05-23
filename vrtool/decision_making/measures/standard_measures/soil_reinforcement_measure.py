import copy
from itertools import product

import numpy as np

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.common_functions import (
    determine_costs,
    determine_new_geometry,
    implement_berm_widening,
)
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.modified_dike_geometry_measure_input import (
    ModifiedDikeGeometryMeasureInput,
)
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class SoilReinforcementMeasure(MeasureProtocol):
    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool = False,
    ):
        # def evaluateMeasure(self, DikeSection, TrajectInfo, preserve_slope=False):
        # To be added: year property to distinguish the same measure in year 2025 and 2045
        # Measure.__init__(self,inputs)
        # self. parameters = measure.parameters
        _measure_type = self.parameters["Type"]

        def get_measure_data(
            modified_measure: ModifiedDikeGeometryMeasureInput,
        ) -> dict:
            _modified_measure = {}
            _modified_measure["id"] = modified_measure.id
            _modified_measure["geometry"] = modified_measure.modified_geometry
            _modified_measure["dcrest"] = modified_measure.d_crest
            _modified_measure["dberm"] = modified_measure.d_berm
            _modified_measure["StabilityScreen"] = self.parameters["StabilityScreen"]
            _modified_measure["l_stab_screen"] = modified_measure.l_stab_screen
            _modified_measure["Cost"] = determine_costs(
                _measure_type,
                dike_section.Length,
                self.unit_costs,
                dcrest=modified_measure.d_crest,
                dberm_in=int(modified_measure.d_house),
                housing=dike_section.houses,
                area_extra=modified_measure.area_extra,
                area_excavated=modified_measure.area_excavated,
                direction=self.parameters["Direction"],
                section=dike_section.name,
            )
            if _modified_measure["StabilityScreen"] == "yes":
                _depth = (
                    dike_section.cover_layer_thickness + modified_measure.l_stab_screen
                )
                _modified_measure["Cost"] += (
                    self.unit_costs.sheetpile * _depth * dike_section.Length
                )
            _modified_measure["Reliability"] = self._get_configured_section_reliability(
                dike_section, traject_info, _modified_measure
            )
            _modified_measure["Reliability"].calculate_section_reliability(
                dike_section.get_cross_sectional_properties()
            )

            return _modified_measure

        modified_dike_geometry_measures = self._get_modified_dike_geometry_measures(
            dike_section
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
        if not isinstance(berm_step, np.ndarray):
            berm_step = np.array(berm_step)

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
            raise ValueError("unknown direction")

    def _get_l_stab_range(self) -> np.ndarray:
        """
        Generates the range of the length of the stability screen,
        relative to the cover thickness

        Returns:
            np.ndarray: list with range for lengths for stability screen;
            without a stability screen a nan is returned
        """
        if self.parameters["StabilityScreen"] == "yes":
            return np.array([3.0, 6.0])
        return np.array([float("nan")])

    def _get_modified_dike_geometry_measures(
        self,
        dike_section: DikeSection,
    ) -> list[ModifiedDikeGeometryMeasureInput]:
        crest_range = self._get_crest_range()
        berm_range = self._get_berm_range()
        l_stab_screen_range = self._get_l_stab_range()

        dike_modifications = list(product(crest_range, berm_range, l_stab_screen_range))

        inputs = []
        for dike_modification in dike_modifications:
            modified_geometry_properties = self._determine_new_geometry(
                dike_section, dike_modification
            )
            measure_input_dictionary = {
                "d_crest": dike_modification[0],
                "d_berm": dike_modification[1],
                "modified_geometry": modified_geometry_properties[0],
                "area_extra": modified_geometry_properties[1],
                "area_excavated": modified_geometry_properties[2],
                "d_house": modified_geometry_properties[3],
                "l_stab_screen": dike_modification[2],
                "id": self.parameters["ID"],
            }

            inputs.append(ModifiedDikeGeometryMeasureInput(**measure_input_dictionary))

        return inputs

    def _determine_new_geometry(
        self,
        dike_section: DikeSection,
        dike_modification: tuple[float],
    ) -> list:
        if hasattr(dike_section, "Kruinhoogte"):
            if dike_section.Kruinhoogte != np.max(dike_section.InitialGeometry.z):
                # In case the crest is unequal to the Kruinhoogte, that value should be given as input as well
                return determine_new_geometry(
                    dike_modification,
                    self.parameters["Direction"],
                    self.parameters["max_outward"],
                    copy.deepcopy(dike_section.InitialGeometry),
                    crest_extra=dike_section.Kruinhoogte,
                )
            else:
                return determine_new_geometry(
                    dike_modification,
                    self.parameters["Direction"],
                    self.parameters["max_outward"],
                    copy.deepcopy(dike_section.InitialGeometry),
                )

        return determine_new_geometry(
            dike_modification,
            self.parameters["Direction"],
            self.parameters["max_outward"],
            copy.deepcopy(dike_section.InitialGeometry),
        )

    def _get_configured_section_reliability(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        modified_geometry_measure: dict,
    ) -> SectionReliability:
        section_reliability = SectionReliability()

        mechanisms = (
            dike_section.section_reliability.failure_mechanisms.get_available_mechanisms()
        )
        for mechanism in mechanisms:
            calc_type = dike_section.mechanism_data[mechanism][0][1]
            mechanism_reliability_collection = (
                self._get_configured_mechanism_reliability_collection(
                    mechanism,
                    calc_type,
                    dike_section,
                    traject_info,
                    modified_geometry_measure,
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
        modified_geometry_measure: dict,
    ) -> MechanismReliabilityCollection:
        mechanism_reliability_collection = MechanismReliabilityCollection(
            mechanism,
            calc_type,
            self.config.T,
            self.config.t_0,
            self.parameters["year"],
        )
        is_first_year_with_widening = True
        for (
            year_to_calculate,
            reliability_input,
        ) in mechanism_reliability_collection.Reliability.items():
            # for all time steps considered.
            # first copy the data
            reliability_input = copy.deepcopy(
                dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                    mechanism
                )
                .Reliability[year_to_calculate]
                .Input
            )
            # Adapt inputs for reliability calculation, but only after year of implementation.
            if float(year_to_calculate) >= self.parameters["year"]:
                _depth_screen = (
                    dike_section.cover_layer_thickness
                    + modified_geometry_measure["l_stab_screen"]
                )
                reliability_input.input = implement_berm_widening(
                    berm_input=reliability_input.input,
                    measure_input=modified_geometry_measure,
                    measure_parameters=self.parameters,
                    mechanism=mechanism,
                    is_first_year_with_widening=is_first_year_with_widening,
                    computation_type=calc_type,
                    path_intermediate_stix=self.config.output_directory
                    / "intermediate_result",
                    depth_screen=_depth_screen,
                )
                is_first_year_with_widening = False
            # put them back in the object
            mechanism_reliability_collection.Reliability[
                year_to_calculate
            ].Input = reliability_input

        mechanism_reliability_collection.generate_LCR_profile(
            dike_section.section_reliability.load,
            traject_info=traject_info,
        )

        return mechanism_reliability_collection
