import numpy as np
import pandas as pd

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
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
    mechanisms: list[MechanismEnum]
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
        return MechanismEnum.STABILITY_INNER in self.mechanisms

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
        It also gathers those measures for which availability is set to 0 and removes these from the list of measures.
        """
        for measure in self.measures:
            measure.evaluate_measure(
                dike_section, traject_info, preserve_slope=preserve_slope
            )

    def solutions_to_dataframe(
        self, filtering: bool = False, splitparams: bool = False
    ):
        # write all solutions to one single dataframe:
        years = self.T
        _mechanism_names = list(map(str, self.mechanisms))
        cols_r = pd.MultiIndex.from_product(
            [_mechanism_names + ["Section"], years], names=["base", "year"]
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
                    "l_stab_screen",
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
                for j, _measure in enumerate(measure.measures):
                    measure_in = []
                    reliability_in = []
                    if _normalized_measure_type in [
                        "soil reinforcement",
                        "soil reinforcement with stability screen",
                    ]:
                        if measure.parameters["StabilityScreen"] == "yes":
                            designvars = (
                                measure.measures[j]["dcrest"],
                                measure.measures[j]["dberm"],
                                measure.measures[j]["l_stab_screen"],
                            )
                        else:
                            designvars = (
                                measure.measures[j]["dcrest"],
                                measure.measures[j]["dberm"],
                                -999,
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
                        measure_in.append(designvars[2])
                    else:
                        measure_in.append(designvars)
                    measure_in.append(cost)

                    betas = measure.measures[j]["Reliability"].SectionReliability

                    _mechanism_names = list(map(str, self.mechanisms))
                    for ij in _mechanism_names + ["Section"]:
                        if ij not in betas.index:
                            # If a mechanism has not been computed it is irrelevant so the beta is assumed to be 10.
                            reliability_in.extend([10.0] * len(self.config.T))
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
                _mechanism_names = list(
                    _mechanism.name for _mechanism in self.mechanisms
                )
                for ij in _mechanism_names + ["Section"]:
                    if ij not in betas.index:
                        # If a mechanism has not been computed it is irrelevant so the beta is assumed to be 10.
                        beta.extend([10.0] * len(self.config.T))
                        continue
                    for ijk in betas.loc[ij].values:
                        beta.append(ijk)
                inputs_r.append(beta)

            elif isinstance(measure.measures, MeasureResultCollectionProtocol):
                _mechanism_names = list(
                    _mechanism.name for _mechanism in self.mechanisms
                )
                (
                    _input_values,
                    _beta_values,
                ) = measure.measures.get_measure_output_values(
                    splitparams, _mechanism_names + ["Section"]
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
