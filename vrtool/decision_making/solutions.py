import copy
import logging

import numpy as np
import pandas as pd
import seaborn as sns

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection


class Solutions:
    # This class contains possible solutions/measures
    section_name: str
    length: float
    initial_geometry: pd.DataFrame
    config: VrtoolConfig
    T: list[int]
    measures: list[MeasureProtocol]
    measure_table: pd.DataFrame

    def __init__(self, dike_section: DikeSection, config: VrtoolConfig):
        self.section_name = dike_section.name
        self.length = dike_section.Length
        self.initial_geometry = dike_section.InitialGeometry

        self.config = config
        self.T = config.T
        self.mechanisms = config.mechanisms
        self.measures: list[MeasureProtocol] = []
        self.measure_table = pd.DataFrame(columns=["ID", "Name"])

    def _is_stability_screen_measure_valid(self) -> bool:
        return "StabilityInner" in self.mechanisms

    def _is_soil_reinforcement_measure_valid(self, stability_screen: str) -> bool:
        if stability_screen.lower().strip() == "yes":
            return self._is_stability_screen_measure_valid()

        return True

    def evaluate_solutions(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool,
    ):
        """This is the base routine to evaluate (i.e., determine costs and reliability) for each defined measure.
        It also gathers those measures for which availability is set to 0 and removes these from the list of measures."""
        for measure in self.measures:
            measure.evaluate_measure(
                dike_section, traject_info, preserve_slope=preserve_slope
            )

    def solutions_to_dataframe(
        self, filtering: bool = False, splitparams: bool = False
    ):
        # write all solutions to one single dataframe:
        years = self.T
        cols_r = pd.MultiIndex.from_product(
            [self.mechanisms + ["Section"], years], names=["base", "year"]
        )
        reliability = pd.DataFrame(columns=cols_r)
        if splitparams:
            cols_m = pd.Index(
                [
                    "ID",
                    "type",
                    "class",
                    "year",
                    "yes/no",
                    "dcrest",
                    "dberm",
                    "beta_target",
                    "transition_level",
                    "cost",
                ],
                name="base",
            )
        else:
            cols_m = pd.Index(
                ["ID", "type", "class", "year", "params", "cost"], name="base"
            )
        measure_df = pd.DataFrame(columns=cols_m)
        # data = pd.DataFrame(columns = cols)
        inputs_m = []
        inputs_r = []

        for measure in self.measures:
            _measure_type = measure.parameters["Type"]
            _normalized_measure_type = _measure_type.lower().strip()
            if isinstance(measure.measures, list):
                # TODO: Deprecated, implement MeasureResultCollectionProtocol for these measures!
                # if it is a list of measures (for soil reinforcement): write each entry of the list to the dataframe
                for j in range(len(measure.measures)):
                    measure_in = []
                    reliability_in = []
                    if _normalized_measure_type in [
                        "soil reinforcement",
                        "soil reinforcement with stability screen",
                    ]:
                        designvars = (
                            measure.measures[j]["dcrest"],
                            measure.measures[j]["dberm"],
                        )

                    cost = measure.measures[j]["Cost"]
                    measure_in.append(str(measure.parameters["ID"]))
                    measure_in.append(_measure_type)
                    measure_in.append(measure.parameters["Class"])
                    measure_in.append(measure.parameters["year"])
                    if splitparams:
                        measure_in.append(-999)
                        measure_in.append(designvars[0])
                        measure_in.append(designvars[1])
                        measure_in.append(-999)
                        measure_in.append(-999)
                    else:
                        measure_in.append(designvars)
                    measure_in.append(cost)

                    betas = measure.measures[j]["Reliability"].SectionReliability

                    for ij in self.mechanisms + ["Section"]:
                        if ij not in betas.index:
                            # TODO (VRTOOL-187).
                            # It seems the other mechanisms are not including Revetment in their measure calculations, therefore failing.
                            # This could happen in the future for other 'new' mechanisms.
                            reliability_in.extend([-999] * len(self.config.T))
                            logging.warning(
                                "Measure '{}' does not contain data for mechanism '{}', using 'nan' instead.".format(
                                    measure.parameters["Name"], ij
                                )
                            )
                            continue
                        for ijk in betas.loc[ij].values:
                            reliability_in.append(ijk)

                    inputs_m.append(measure_in)
                    inputs_r.append(reliability_in)

            elif isinstance(measure.measures, dict):
                # TODO: Deprecated, implement MeasureResultCollectionProtocol for these measures!
                ID = str(measure.parameters["ID"])
                if _normalized_measure_type == "vertical geotextile":
                    designvars = measure.measures["VZG"]

                if _normalized_measure_type == "diaphragm wall":
                    designvars = measure.measures["DiaphragmWall"]

                if _normalized_measure_type == "revetment":
                    designvars = measure.measures["Revetment"]

                if _normalized_measure_type == "custom":
                    designvars = 1.0  ##TODO check

                measure_class = measure.parameters["Class"]
                year = measure.parameters["year"]
                cost = measure.measures["Cost"]
                if splitparams:
                    inputs_m.append(
                        [
                            ID,
                            _measure_type,
                            measure_class,
                            year,
                            designvars,
                            -999,
                            -999,
                            -999,
                            -999,
                            cost,
                        ]
                    )
                else:
                    inputs_m.append(
                        [
                            ID,
                            _measure_type,
                            measure_class,
                            year,
                            designvars,
                            cost,
                        ]
                    )
                betas = measure.measures["Reliability"].SectionReliability
                beta = []
                for ij in self.mechanisms + ["Section"]:
                    if ij not in betas.index:
                        # TODO (VRTOOL-187).
                        # It seems the other mechanisms are not including Revetment in their measure calculations, therefore failing.
                        # This could happen in the future for other 'new' mechanisms.
                        beta.extend([-999] * len(self.config.T))
                        logging.warning(
                            "Measure '{}' does not contain data for mechanism '{}', using 'nan' instead.".format(
                                measure.parameters["Name"], ij
                            )
                        )
                        continue
                    for ijk in betas.loc[ij].values:
                        beta.append(ijk)
                inputs_r.append(beta)

            elif isinstance(measure.measures, MeasureResultCollectionProtocol):
                (
                    _input_values,
                    _beta_values,
                ) = measure.measures.get_measure_output_values(
                    splitparams, self.mechanisms + ["Section"]
                )
                inputs_m.extend(_input_values)
                inputs_r.extend(_beta_values)

        reliability = pd.concat((reliability, pd.DataFrame(inputs_r, columns=cols_r)))
        measure_df = pd.concat((measure_df, pd.DataFrame(inputs_m, columns=cols_m)))
        cols = pd.MultiIndex.from_arrays(
            np.array([measure_df.columns, [""] * len(measure_df.columns)])
        )
        measure_df.columns = cols
        self.MeasureData = measure_df.join(reliability, how="inner")
        if (
            filtering
        ):  # here we could add some filtering on the measures, but it is not used right now.
            pass
