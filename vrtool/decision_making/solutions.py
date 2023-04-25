import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures import (
    CustomMeasure,
    DiaphragmWallMeasure,
    SoilReinforcementMeasure,
    StabilityScreenMeasure,
    VerticalGeotextileMeasure,
)
from vrtool.decision_making.measures.measure_base import MeasureBase
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection

import logging


class Solutions:
    # This class contains possible solutions/measures
    def __init__(self, dike_section: DikeSection, config: VrtoolConfig):
        self.section_name = dike_section.name
        self.length = dike_section.Length
        self.initial_geometry = dike_section.InitialGeometry

        self.config = config
        self.T = config.T
        self.mechanisms = config.mechanisms
        self.measures: list[MeasureBase] = []

    def fillSolutions(self, excel_sheet: Path):
        """This routine reads input for the measures from the Excel sheet for each section.
        It identifies combinables and partials and identifies possible combinations of measures this way.
        These are then stored in the MeasureTable, which is later evaluated.
        """
        data = pd.read_excel(excel_sheet, "Measures")
        combinables = []
        partials = []
        for i in data.index:
            # TODO depending on data.loc[i].type make correct sublclass

            loc_data = data.loc[i]
            measure_type = data.loc[i].Type
            if measure_type == "Soil reinforcement":
                if not self._is_soil_reinforcement_measure_valid(
                    loc_data.StabilityScreen
                ):
                    logging.warn(
                        f'No stability inner mechanism present for soil reinforcement with stability screen measure "{loc_data.ID}" in "{excel_sheet.stem}."'
                    )
                else:
                    self.measures.append(
                        SoilReinforcementMeasure(data.loc[i], self.config)
                    )
            elif measure_type == "Diaphragm Wall":
                self.measures.append(DiaphragmWallMeasure(data.loc[i], self.config))
            elif measure_type == "Stability Screen":
                if not self._is_stability_screen_measure_valid():
                    logging.warn(
                        f'No stability inner mechanism present for stability screen measure "{loc_data.ID}" in "{excel_sheet.stem}."'
                    )
                else:
                    self.measures.append(
                        StabilityScreenMeasure(data.loc[i], self.config)
                    )
            elif measure_type == "Vertical Geotextile":
                self.measures.append(
                    VerticalGeotextileMeasure(data.loc[i], self.config)
                )
            elif data.loc[i].Type == "Custom":
                data.loc[i, "File"] = excel_sheet.parent.joinpath(
                    "Measures", data.loc[i]["File"]
                )
                self.measures.append(CustomMeasure(data.loc[i], self.config))

        self.measure_table = pd.DataFrame(columns=["ID", "Name"])
        for i, measure in enumerate(self.measures):
            if measure.parameters["available"] == 1:
                self.measure_table.loc[i] = [
                    str(measure.parameters["ID"]),
                    measure.parameters["Name"],
                ]
                # also add the potential combined solutions up front
                if measure.parameters["Class"] == "combinable":
                    combinables.append(
                        (measure.parameters["ID"], measure.parameters["Name"])
                    )
                if measure.parameters["Class"] == "partial":
                    partials.append(
                        (measure.parameters["ID"], measure.parameters["Name"])
                    )
        count = 0
        for i in range(0, len(partials)):
            for j in range(0, len(combinables)):
                self.measure_table.loc[count + len(self.measures) + 1] = [
                    str(partials[i][0]) + "+" + str(combinables[j][0]),
                    str(partials[i][1]) + "+" + str(combinables[j][1]),
                ]
                count += 1

    def _is_stability_screen_measure_valid(self) -> bool:
        return "StabilityInner" in self.mechanisms

    def _is_soil_reinforcement_measure_valid(self, stability_screen: str) -> bool:
        if stability_screen == "yes":
            return self._is_stability_screen_measure_valid()

        return True

    def evaluate_solutions(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool = False,
    ):
        """This is the base routine to evaluate (i.e., determine costs and reliability) for each defined measure.
        It also gathers those measures for which availability is set to 0 and removes these from the list of measures."""
        self.trange = self.T
        removal = []
        for i, measure in enumerate(self.measures):
            if measure.parameters["available"] == 1:
                # old: measure.evaluateMeasure(DikeSection, TrajectInfo, preserve_slope = preserve_slope)
                measure.evaluate_measure(
                    dike_section, traject_info, preserve_slope=preserve_slope
                )

                # if measure.parameters['Type'] == 'Soil reinforcement':
                #     A = Soilreinforcement()
                #     A.evaluateMeasure(measure, DikeSection, TrajectInfo, preserve_slope = preserve_slope)
                # elif measure.parameters['Type'] == 'DiaphragmWall':
                #     DiaphragmWall.evaluateMeasure(measure,DikeSection, TrajectInfo)
                # elif measure.parameters['Type'] == 'StabilityScreen':
                #     StabilityScreen.evaluateMeasure(measure,DikeSection, TrajectInfo)
                # elif measure.parameters['Type'] == 'VerticalGeotextile':
                #     VerticalGeotextile.evaluateMeasure(measure,DikeSection, TrajectInfo)
                # elif measure.parameters['Type'] == 'Custom':
                #     CustomMeasure.evaluateMeasure(measure)
            else:
                removal.append(i)
        # remove measures that are set to unavailable:
        if len(removal) > 0:
            for i in reversed(removal):
                self.measures.pop(i)

    def solutions_to_dataframe(self, filtering=False, splitparams=False):
        # write all solutions to one single dataframe:

        years = self.trange
        cols_r = pd.MultiIndex.from_product(
            [self.mechanisms + ["Section"], years], names=["base", "year"]
        )
        reliability = pd.DataFrame(columns=cols_r)
        if splitparams:
            cols_m = pd.Index(
                ["ID", "type", "class", "year", "yes/no", "dcrest", "dberm", "cost"],
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

        for i, measure in enumerate(self.measures):
            if isinstance(measure.measures, list):
                # if it is a list of measures (for soil reinforcement): write each entry of the list to the dataframe
                type = measure.parameters["Type"]

                for j in range(len(measure.measures)):
                    measure_in = []
                    reliability_in = []
                    if type == "Soil reinforcement":
                        designvars = (
                            measure.measures[j]["dcrest"],
                            measure.measures[j]["dberm"],
                        )

                    cost = measure.measures[j]["Cost"]
                    measure_in.append(str(measure.parameters["ID"]))
                    measure_in.append(type)
                    measure_in.append(measure.parameters["Class"])
                    measure_in.append(measure.parameters["year"])
                    if splitparams:
                        measure_in.append(-999)
                        measure_in.append(designvars[0])
                        measure_in.append(designvars[1])
                    else:
                        measure_in.append(designvars)
                    measure_in.append(cost)

                    betas = measure.measures[j]["Reliability"].SectionReliability

                    for ij in self.mechanisms + ["Section"]:
                        for ijk in betas.loc[ij].values:
                            reliability_in.append(ijk)

                    inputs_m.append(measure_in)
                    inputs_r.append(reliability_in)

            elif isinstance(measure.measures, dict):
                ID = str(measure.parameters["ID"])
                type = measure.parameters["Type"]
                if type == "Vertical Geotextile":
                    designvars = measure.measures["VZG"]

                if type == "Diaphragm Wall":
                    designvars = measure.measures["DiaphragmWall"]

                if type == "Custom":
                    designvars = 1.0  ##TODO check

                measure_class = measure.parameters["Class"]
                year = measure.parameters["year"]
                cost = measure.measures["Cost"]
                if splitparams:
                    inputs_m.append(
                        [ID, type, measure_class, year, designvars, -999, -999, cost]
                    )
                else:
                    inputs_m.append([ID, type, measure_class, year, designvars, cost])
                betas = measure.measures["Reliability"].SectionReliability
                beta = []
                for ij in self.mechanisms + ["Section"]:
                    for ijk in betas.loc[ij].values:
                        beta.append(ijk)
                inputs_r.append(beta)
        # reliability = reliability.append(pd.DataFrame(inputs_r, columns=cols_r))
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

    def plot_beta_time_euro(
        self,
        measures="undefined",
        mechanism="Section",
        beta_ind="beta0",
        sectionname="Unknown",
        beta_req=None,
    ):
        # This function plots the relation between cost and beta in a certain year

        # measures is a list of measures that need to be plotted
        if measures == "undefined":
            measures = list(self.measures)

        # mechanism can be used to select a single or all ('Section') mechanisms
        # beta can be used to use a criterion for selecting the 'best' designs, such as the beta at 't0'
        cols = ["type", "parameters", "Cost"]
        [cols.append("beta" + str(i)) for i in self.trange]
        data = pd.DataFrame(columns=cols)
        num_plots = 5
        colors = sns.color_palette("hls", n_colors=num_plots)
        # colors = plt.cm.get_cmap(name=plt.cm.hsv, lut=num_plots)
        color = 0

        for i in np.unique(self.MeasureData["ID"].values):
            if isinstance(self.measures[int(i) - 1].measures, list):
                data = copy.deepcopy(self.MeasureData.loc[self.MeasureData["ID"] == i])
                # inputs = []; type = self.Measures[i].parameters['Type']
                # for j in range(0, len(self.Measures[i].measures)):
                #     inputvals = []
                #     if type == 'Soil reinforcement': designvars = str((self.Measures[i].measures[j]['dcrest'], self.Measures[i].measures[j]['dberm']))
                #     betas = list(self.Measures[i].measures[j]['Reliability'].SectionReliability.loc[mechanism])
                #     cost = self.Measures[i].measures[j]['Cost']
                #     inputvals.append(type); inputvals.append(designvars); inputvals.append(cost)
                #     for ij in range(0, len(betas)): inputvals.append(betas[ij])
                #     inputs.append(inputvals)
                # data = data.append(pd.DataFrame(inputs, columns=cols))
                # x = data.loc[data['type'] == 'Soil reinforcement']
                y = copy.deepcopy(data)
                x = data.sort_values(by=["cost"])

                steps = 20
                cost_grid = np.linspace(np.min(x["cost"]), np.max(x["cost"]), steps)
                envelope_beta = []
                envelope_costs = []
                indices = []
                betamax = 0

                for j in range(len(cost_grid) - 1):
                    values = x.loc[
                        (x["cost"] >= (cost_grid[j]))
                        & (x["cost"] <= (cost_grid[j + 1]))
                    ][(mechanism, beta_ind)]
                    if len(list(values)) > 0:
                        idd = values.idxmax()
                        if betamax < np.max(list(values)):
                            betamax = np.max(list(values))
                            indices.append(idd)
                            if isinstance(x["cost"].loc[idd], pd.Series):
                                envelope_costs.append(x["cost"].loc[idd].values[0])

                            if not isinstance(x["cost"].loc[idd], pd.Series):
                                envelope_costs.append(x["cost"].loc[idd])

                            envelope_beta.append(betamax)

                if self.measures[np.int_(i) - 1].parameters["Name"][-4:] != "2045":
                    plt.plot(
                        envelope_costs,
                        envelope_beta,
                        color=colors[color],
                        linestyle="-",
                    )
                    # [plt.text(y['Cost'].loc[ij], y[beta_ind].loc[i], y['parameters'].loc[ij],fontsize='x-small') for ij in indices]

                    plt.plot(
                        y["cost"],
                        y[(mechanism, beta_ind)],
                        label=self.measures[np.int_(i) - 1].parameters["Name"],
                        marker="o",
                        markersize=6,
                        color=colors[color],
                        markerfacecolor=colors[color],
                        markeredgecolor=colors[color],
                        linestyle="",
                        alpha=1,
                    )

                    color += 1
            elif isinstance(self.measures[np.int_(i) - 1].measures, dict):
                data = copy.deepcopy(self.MeasureData.loc[self.MeasureData["ID"] == i])
                plt.plot(
                    data["cost"],
                    data[(mechanism, beta_ind)],
                    label=self.measures[np.int_(i) - 1].parameters["Name"],
                    marker="d",
                    markersize=10,
                    markerfacecolor=colors[color],
                    markeredgecolor=colors[color],
                    linestyle="",
                )
                color += 1
        axes = plt.gca()
        plt.plot([0, axes.get_xlim()[1]], [beta_req, beta_req], "k--", label="Norm")
        plt.xlabel("Cost")
        plt.ylabel(r"$\beta_{" + str(beta_ind + 2025) + "}$")
        plt.title("Cost-beta relation for " + mechanism + " at " + sectionname)
        plt.legend(loc="best")
