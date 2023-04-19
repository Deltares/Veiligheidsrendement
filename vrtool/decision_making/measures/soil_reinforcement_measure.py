import copy
import logging

import numpy as np

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


class SoilReinforcementMeasure(MeasureBase):
    # type == 'Soil reinforcement':
    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: dict[str, any],
        plot_dir: bool = False,
        preserve_slope: bool = False,
    ):
        # def evaluateMeasure(self, DikeSection, TrajectInfo, preserve_slope=False):
        # To be added: year property to distinguish the same measure in year 2025 and 2045
        # Measure.__init__(self,inputs)
        # self. parameters = measure.parameters

        SFincrease = 0.2  # for stability screen

        type = self.parameters["Type"]
        mechanisms = dike_section.section_reliability.Mechanisms.keys()
        crest_step = self.crest_step
        berm_step = self.berm_step
        crestrange = np.linspace(
            self.parameters["dcrest_min"],
            self.parameters["dcrest_max"],
            np.int_(
                1
                + (self.parameters["dcrest_max"] - self.parameters["dcrest_min"])
                / crest_step
            ),
        )
        # TODO: CLEAN UP, make distinction between inwards and outwards, so xin, xout and y,and adapt DetermineNewGeometry
        if self.parameters["Direction"] == "outward":
            if np.size(berm_step) > 1:
                max_berm = (
                    self.parameters["max_outward"] + self.parameters["max_inward"]
                )
                bermrange = berm_step[: len(np.where((berm_step <= max_berm))[0])]
            else:
                bermrange = np.linspace(
                    0.0,
                    self.parameters["max_outward"],
                    np.int_(1 + (self.parameters["max_outward"] / berm_step)),
                )
        elif self.parameters["Direction"] == "inward":
            if np.size(berm_step) > 1:
                max_berm = self.parameters["max_inward"]
                bermrange = berm_step[: len(np.where((berm_step <= max_berm))[0])]
            else:
                bermrange = np.linspace(
                    0.0,
                    self.parameters["max_inward"],
                    np.int_(1 + (self.parameters["max_inward"] / berm_step)),
                )
        else:
            raise Exception("unkown direction")

        measures = [[x, y] for x in crestrange for y in bermrange]
        if not preserve_slope:
            slope_in = 4
            slope_out = 3  # inner and outer slope
        else:
            slope_in = False
            slope_out = False

        self.measures = []
        if self.parameters["StabilityScreen"] == "yes":
            d_cover_input = (
                dike_section.section_reliability.Mechanisms["StabilityInner"]
                .Reliability["0"]
                .Input.input.get("d_cover", None)
            )
            if d_cover_input:
                if d_cover_input.size > 1:
                    logging.info("d_cover has more values than 1.")

                self.parameters["Depth"] = max([d_cover_input[0] + 1.0, 8.0])
            else:
                self.parameters[
                    "Depth"
                ] = 6.0  # TODO: implement a better depth estimate based on d_cover

        for j in measures:
            if self.parameters["Direction"] == "outward":
                k = max(
                    0, j[1] - self.parameters["max_inward"]
                )  # correction max_outward
            else:
                k = j[1]
            self.measures.append({})
            self.measures[-1]["dcrest"] = j[0]
            self.measures[-1]["dberm"] = j[1]
            if hasattr(dike_section, "Kruinhoogte"):
                if dike_section.Kruinhoogte != np.max(dike_section.InitialGeometry.z):
                    # In case the crest is unequal to the Kruinhoogte, that value should be given as input as well
                    (
                        self.measures[-1]["Geometry"],
                        area_extra,
                        area_excavated,
                        dhouse,
                    ) = determine_new_geometry(
                        j,
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
                    (
                        self.measures[-1]["Geometry"],
                        area_extra,
                        area_excavated,
                        dhouse,
                    ) = determine_new_geometry(
                        j,
                        self.parameters["Direction"],
                        self.parameters["max_outward"],
                        copy.deepcopy(dike_section.InitialGeometry),
                        self.geometry_plot,
                        **{"plot_dir": plot_dir, "slope_in": slope_in},
                    )
            else:
                (
                    self.measures[-1]["Geometry"],
                    area_extra,
                    area_excavated,
                    dhouse,
                ) = determine_new_geometry(
                    j,
                    self.parameters["Direction"],
                    self.parameters["max_outward"],
                    copy.deepcopy(dike_section.InitialGeometry),
                    self.geometry_plot,
                    **{"plot_dir": plot_dir, "slope_in": slope_in},
                )

            self.measures[-1]["Cost"] = determine_costs(
                self.parameters,
                type,
                dike_section.Length,
                self.unit_costs,
                dcrest=j[0],
                dberm_in=int(dhouse),
                housing=dike_section.houses,
                area_extra=area_extra,
                area_excavated=area_excavated,
                direction=self.parameters["Direction"],
                section=dike_section.name,
            )
            self.measures[-1]["Reliability"] = SectionReliability()
            self.measures[-1]["Reliability"].Mechanisms = {}

            for i in mechanisms:
                calc_type = dike_section.mechanism_data[i][1]
                self.measures[-1]["Reliability"].Mechanisms[
                    i
                ] = MechanismReliabilityCollection(
                    i, calc_type, self.config, measure_year=self.parameters["year"]
                )
                for ij, reliability_input in (
                    self.measures[-1]["Reliability"].Mechanisms[i].Reliability.items()
                ):
                    # for all time steps considered.
                    # first copy the data
                    reliability_input = copy.deepcopy(
                        dike_section.section_reliability.Mechanisms[i]
                        .Reliability[ij]
                        .Input
                    )
                    # Adapt inputs for reliability calculation, but only after year of implementation.
                    if float(ij) >= self.parameters["year"]:
                        reliability_input.input = implement_berm_widening(
                            input=reliability_input.input,
                            measure_input=self.measures[-1],
                            measure_parameters=self.parameters,
                            mechanism=i,
                            computation_type=calc_type,
                        )
                    # put them back in the object
                    self.measures[-1]["Reliability"].Mechanisms[i].Reliability[
                        ij
                    ].Input = reliability_input
                self.measures[-1]["Reliability"].Mechanisms[i].generateLCRProfile(
                    dike_section.section_reliability.Load,
                    mechanism=i,
                    trajectinfo=traject_info,
                )
            self.measures[-1]["Reliability"].calculate_section_reliability()
