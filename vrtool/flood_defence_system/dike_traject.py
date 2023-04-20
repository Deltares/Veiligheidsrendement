from __future__ import annotations

import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


class DikeTraject:
    sections: list[DikeSection]
    general_info: DikeTrajectInfo
    probabilities: pd.DataFrame

    # This class contains general information on the dike traject and is used to store all data on the sections
    def __init__(self, config: VrtoolConfig, traject=None):
        if traject == None:
            logging.warn("No traject given in config. Default was chosen")
            self.traject = "Not specified"
        else:
            self.traject = traject

        self.mechanism_names = config.mechanisms
        self.assessment_plot_years = config.assessment_plot_years
        self.flip_traject = config.flip_traject
        self.t_0 = config.t_0
        self.T = config.T

        self.sections = DikeSection.get_dike_sections_from_vr_config(config)

        traject_length = sum(map(lambda x: x.Length, self.sections))
        self.general_info = DikeTrajectInfo.from_traject_info(
            config.traject, traject_length
        )

    def set_probabilities(self):
        """routine to make 1 dataframe of all probabilities of a TrajectObject"""
        for i, section in enumerate(self.sections):
            if i == 0:
                _assessment = (
                    section.section_reliability.SectionReliability.reset_index()
                )
                _assessment["Section"] = section.name
                _assessment["Length"] = section.Length
                _assessment.columns = _assessment.columns.astype(str)
                if "mechanism" in _assessment.columns:
                    _assessment = _assessment.rename(columns={"mechanism": "index"})
            else:
                data_to_add = (
                    section.section_reliability.SectionReliability.reset_index()
                )
                data_to_add["Section"] = section.name
                data_to_add["Length"] = section.Length
                data_to_add.columns = data_to_add.columns.astype(str)
                if "mechanism" in data_to_add.columns:
                    data_to_add = data_to_add.rename(columns={"mechanism": "index"})

                _assessment = pd.concat((_assessment, data_to_add))
        _assessment = _assessment.rename(
            columns={"index": "mechanism", "Section": "name"}
        )
        self.probabilities = _assessment.reset_index(drop=True).set_index(
            ["name", "mechanism"]
        )

    def plot_assessment(
        self,
        fig_size=(6, 4),
        draw_targetbeta="off",
        last=True,
        alpha=1,
        colors=False,
        labels_limited=False,
        system_rel=False,
        custom_name=False,
        title_in=False,
        reinforcement_strategy=False,
        greedymode="Optimal",
        show_xticks=True,
        t_list=[],
        case_settings={"directory": Path(""), "language": "NL", "beta_or_prob": "beta"},
    ):
        """Routine to plot traject reliability"""
        if reinforcement_strategy:
            if reinforcement_strategy.__class__.__name__ == "GreedyStrategy":
                if greedymode == "Optimal":
                    ProbabilityFrame = reinforcement_strategy.Probabilities[
                        reinforcement_strategy.OptimalStep
                    ]
                elif greedymode == "SatisfiedStandard":
                    Ptarget = self.general_info.Pmax
                    for i in reversed(reinforcement_strategy.Probabilities):
                        beta_traj, Pf_traj = calc_traject_prob(i, ts=50)
                        if Pf_traj < Ptarget:  # satisfactory solution
                            ProbabilityFrame = i
                        else:
                            if not "ProbabilityFrame" in locals():
                                logging.warn(
                                    "No satisfactory solution found, skipping plot"
                                )
                            return
            else:
                ProbabilityFrame = reinforcement_strategy.Probabilities[-1]
        else:
            ProbabilityFrame = self.probabilities
            ProbabilityFrame = ProbabilityFrame.drop(["Length"], axis=1)
        ProbabilityFrame.columns = ProbabilityFrame.columns.values.astype(np.int64)
        plot_settings()

        self.probabilities.to_csv(
            case_settings["directory"].joinpath("InitialAssessment_Betas.csv")
        )
        # English or Dutch labels and titles
        if case_settings["language"] == "NL":
            label_xlabel = "Dijkvakken"
            if case_settings["beta_or_prob"] == "beta":
                label_ylabel = r"Betrouwbaarheidsindex $\beta$ [-/jaar]"
                label_target = "Doelbetrouwbaarheid"
            elif case_settings["beta_or_prob"] == "prob":
                label_ylabel = r"Faalkans $P_f$ [-/jaar]"
                label_target = "Doelfaalkans"
            labels_xticks = []
            for i in self.sections:
                labels_xticks.append(i.name)
        elif case_settings["language"] == "EN":
            label_xlabel = "Dike sections"
            if case_settings["beta_or_prob"] == "beta":
                if labels_limited:
                    label_ylabel = r"$\beta$ [-/year]"
                else:
                    label_ylabel = r"Reliability index $\beta$ [-/year]"
                label_target = r"$\beta_\mathrm{target}$"
            elif case_settings["beta_or_prob"] == "prob":
                label_ylabel = r"Failure probability $P_f$ [-/year]"
                label_target = "Target failure prob."
            labels_xticks = []
            for i in self.sections:
                labels_xticks.append("S" + i.name[2:])

        cumlength, xticks1, middles = get_section_length_in_traject(
            self.probabilities["Length"]
            .loc[self.probabilities.index.get_level_values(1) == "Overflow"]
            .values
        )

        if colors:
            color = sns.cubehelix_palette(**colors)
        else:
            color = sns.cubehelix_palette(
                n_colors=4, start=1.9, rot=1, gamma=1.5, hue=1.0, light=0.8, dark=0.3
            )
        # color = sns.cubehelix_palette(n_colors=4, start=0.7,rot=1,gamma=1.5,hue=0.0,light=0.8,dark=0.3)
        markers = ["o", "v", "d"]

        # We will make plots for different years
        year = 0
        line = {}
        mid = {}
        legend_line = {}
        if len(t_list) == 0:
            t_list = self.assessment_plot_years
        for ii in t_list:
            if system_rel:
                fig, (ax, ax1) = plt.subplots(
                    nrows=1,
                    ncols=2,
                    figsize=fig_size,
                    sharey="row",
                    gridspec_kw={
                        "width_ratios": [20, 1],
                        "wspace": 0.08,
                        "left": 0.03,
                        "right": 0.98,
                    },
                )
            else:
                fig, ax = plt.subplots(figsize=fig_size)
            col = 0
            mech = 0
            for j in self.mechanism_names:
                # get data to plot
                # plotdata = self.Probabilities[str(ii)].loc[self.Probabilities['index'] == j].values
                plotdata = (
                    ProbabilityFrame[ii]
                    .loc[ProbabilityFrame.index.get_level_values(1) == j]
                    .values
                )
                if case_settings["beta_or_prob"] == "prob":
                    plotdata = beta_to_pf(plotdata)
                ydata = copy.deepcopy(plotdata)
                for ij in range(0, len(plotdata)):
                    ydata = np.insert(ydata, ij * 2, plotdata[ij])

                if year < 1000:  # year == 0:
                    # define the lines for the first time. Else replace the data.
                    (line[mech],) = ax.plot(
                        xticks1, ydata, color=color[col], linestyle="-", alpha=alpha
                    )
                    (mid[mech],) = ax.plot(
                        middles,
                        plotdata,
                        color=color[col],
                        linestyle="",
                        marker=markers[col],
                        alpha=alpha,
                    )
                    (legend_line[mech],) = ax.plot(
                        -999,
                        -999,
                        color=color[col],
                        linestyle="-",
                        marker=markers[col],
                        alpha=alpha,
                        label=j,
                    )
                else:
                    line[mech].set_ydata(ydata)
                    mid[mech].set_ydata(plotdata)
                col += 1
                mech += 1
            if system_rel:
                (legend_line[mech],) = ax.plot(
                    -999,
                    -999,
                    color=color[col],
                    linestyle="-",
                    alpha=alpha,
                    label="System",
                )
            col = 0
            # Whether to draw the target reliability for each individula mechanism.
            if draw_targetbeta == "on" and last:
                for j in self.mechanism_names:
                    dash = [2, 2]
                    if j == "StabilityInner":
                        N = (
                            self.general_info.TrajectLength
                            * self.general_info.aStabilityInner
                            / self.general_info.bStabilityInner
                        )
                        pt = (
                            self.general_info.Pmax
                            * self.general_info.omegaStabilityInner
                            / N
                        )
                        # dash = [1,2]
                    elif j == "Piping":
                        N = (
                            self.general_info.TrajectLength
                            * self.general_info.aPiping
                            / self.general_info.bPiping
                        )
                        pt = self.general_info.Pmax * self.general_info.omegaPiping / N
                        # dash = [1,3]
                    elif j == "Overflow":
                        pt = self.general_info.Pmax * self.general_info.omegaOverflow
                        # dash = [1,2]
                    if case_settings["beta_or_prob"] == "beta":
                        ax.plot(
                            [0, max(cumlength)],
                            [
                                pf_to_beta(pt),
                                pf_to_beta(pt),
                            ],
                            color=color[col],
                            linestyle=":",
                            label=label_target + " " + j,
                            dashes=dash,
                            alpha=0.5,
                            linewidth=1,
                        )
                    elif case_settings["beta_or_prob"] == "prob":
                        ax.plot(
                            [0, max(cumlength)],
                            [pt, pt],
                            color=color[col],
                            linestyle=":",
                            label=label_target + " " + j,
                            dashes=dash,
                            alpha=0.5,
                            linewidth=1,
                        )
                    col += 1
            if last:
                for i in cumlength:
                    ax.axvline(
                        x=i, color="gray", linestyle="-", linewidth=0.5, alpha=0.5
                    )
                if case_settings["beta_or_prob"] == "beta":
                    # should be in legend
                    ax.plot(
                        [0, max(cumlength)],
                        [
                            pf_to_beta(self.general_info.Pmax),
                            pf_to_beta(self.general_info.Pmax),
                        ],
                        "k--",
                        label=label_target,
                        linewidth=1,
                    )
                if case_settings["beta_or_prob"] == "prob":
                    ax.plot(
                        [0, max(cumlength)],
                        [self.general_info.Pmax, self.general_info.Pmax],
                        "k--",
                        label=label_target,
                        linewidth=1,
                    )

                ax.legend(loc=1)
                if not labels_limited:
                    ax.set_xlabel(label_xlabel)
                ax.set_ylabel(label_ylabel)
                ax.set_xticks(middles)
                if show_xticks:
                    ax.set_xticklabels(labels_xticks)
                else:
                    ax.set_xticklabels("")
                ax.tick_params(axis="x", rotation=90)
                ax.set_xlim([0, max(cumlength)])
                ax.tick_params(axis="both", bottom=False)
                if case_settings["beta_or_prob"] == "beta":
                    ax.set_ylim([0.5, 8.5])

                if case_settings["beta_or_prob"] == "prob":
                    ax.set_ylim([1e-1, 1e-9])
                    ax.set_yscale("log")

                ax.grid(axis="y", linewidth=0.5, color="gray", alpha=0.5)

                if self.flip_traject:
                    ax.invert_xaxis()
            if system_rel:
                col = 0
                mech = 0
                line1 = {}
                mid1 = {}
                bars = {}
                pt_tot = 0
                for m in self.mechanism_names:
                    beta_t, p_t = calc_traject_prob(ProbabilityFrame, ts=ii, mechs=[m])
                    # pt_tot +=p_t
                    pt_tot = 1 - ((1 - pt_tot) * (1 - p_t))
                    # line1[mech], = ax1.plot([0,1], [beta_t,beta_t], color=color[col], linestyle='-', label=j, alpha=alpha)
                    # mid1[mech], = ax1.plot(0.5, beta_t, color=color[col], linestyle='', marker=markers[col], alpha=alpha)
                    bars[mech] = ax1.bar(col, beta_t, color=color[col])
                    col += 1
                    mech += 1
                beta_tot = pf_to_beta(pt_tot)
                logging.info(beta_tot)
                ax1.plot([-2, 3], [beta_tot, beta_tot], color=color[col])
                ax1.axhline(
                    pf_to_beta(self.general_info.Pmax),
                    linestyle="--",
                    color="black",
                    label=label_target,
                    linewidth=1,
                )
                ax1.grid(axis="y", linewidth=0.5, color="gray", alpha=0.5)
                ax1.set_xticks([0, 1, 2])
                ax1.set_xlim(left=-0.4, right=2.4)
                ax1.set_title("System \n reliability")
                if show_xticks:
                    ax1.set_xticklabels(self.mechanism_names, rotation=90, fontsize=6)
                else:
                    ax1.set_xticklabels("")
                ax1.tick_params(axis="both", bottom=False)
                if title_in:
                    ax.set_title(title_in)
            if not custom_name:
                custom_name = (
                    case_settings["beta_or_prob"]
                    + "_"
                    + str(self.t_0 + ii)
                    + "_Assessment.png"
                )
            plt.savefig(
                case_settings["directory"].joinpath(custom_name),
                dpi=300,
                bbox_inches="tight",
                format="png",
            )
            plt.close()
            custom_name = False
            year += 1
            if last:
                plt.close()

    def plot_assessment_results(
        self, output_directory: Path, section_ids=None, t_start=2020
    ):
        # for all or a selection of sections:
        if section_ids == None:
            sections = self.sections
        else:
            sections = []
            for i in section_ids:
                sections.append(self.sections[i])

        if not output_directory.exists():
            output_directory.mkdir(parents=True)

        # generate plots
        for i in sections:
            plt.figure(1)
            [
                i.section_reliability.Mechanisms[j].drawLCR(
                    label=j, type="Standard", tstart=t_start
                )
                for j in self.mechanism_names
            ]
            plt.plot(
                [t_start, t_start + np.max(self.T)],
                [
                    pf_to_beta(self.general_info.Pmax),
                    pf_to_beta(self.general_info.Pmax),
                ],
                "k--",
                label="Requirement",
            )
            plt.legend()
            plt.title(i.name)
            plt.savefig(output_directory.joinpath(i.name + ".png"), bbox_inches="tight")
            plt.close()


def plot_settings(labels: str = "NL"):
    # a bunch of settings to make it look nice:
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_section_length_in_traject(length):
    # Derive some coordinates to properly plot everything according to the length of the different sections:
    cumlength = np.cumsum(length)
    cumlength = np.insert(cumlength, 0, 0)
    xticks1 = copy.deepcopy(cumlength)
    for i in range(1, len(cumlength) - 1):
        xticks1 = np.insert(xticks1, i * 2, cumlength[i])
    middles = (cumlength[:-1] + cumlength[1:]) / 2
    return cumlength, xticks1, middles


def calc_traject_prob(base, horizon=False, datatype="DataFrame", ts=None, mechs=False):
    pfs = {}
    if horizon:
        trange = np.arange(0, horizon, 1)
    elif ts != None:
        trange = [ts]
    else:
        raise ValueError("No range defined")
    if datatype == "DataFrame":
        ts = base.columns.values
        if not mechs:
            mechs = np.unique(base.index.get_level_values("mechanism").values)
        # mechs = ['Overflow']
    # pf_traject = np.zeros((len(ts),))
    pf_traject = np.zeros((len(trange),))

    for i in mechs:
        if i != "Section":
            if datatype == "DataFrame":
                betas = base.xs(i, level="mechanism").values.astype("float")
            else:
                betas = base[i]
            beta_interp = interp1d(np.array(ts).astype(np.int_), betas)
            pfs[i] = beta_to_pf(beta_interp(trange))
            # pfs[i] = ProbabilisticFunctions.beta_to_pf(betas)
            pnonfs = 1 - pfs[i]
            if i == "Overflow":
                # pf_traject += np.max(pfs[i], axis=0)
                pf_traject = 1 - np.multiply(1 - pf_traject, 1 - np.max(pfs[i], axis=0))
            else:
                # pf_traject += np.sum(pfs[i], axis=0)
                # pf_traject += 1-np.prod(pnonfs, axis=0)
                pf_traject = 1 - np.multiply(1 - pf_traject, np.prod(pnonfs, axis=0))

    ## INTERPOLATION AFTER COMBINATION:
    # pfail = interp1d(ts,pf_traject)
    # p_t1 = ProbabilisticFunctions.beta_to_pf(pfail(trange))
    # betafail = interp1d(ts, ProbabilisticFunctions.pf_to_beta(pf_traject),kind='linear')
    # beta_t = betafail(trange)
    # p_t = ProbabilisticFunctions.beta_to_pf(np.array(beta_t, dtype=np.float64))

    beta_t = pf_to_beta(pf_traject)
    p_t = pf_traject
    return beta_t, p_t
