
import copy
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.strategy_base import StrategyBase

import itertools
from tools.HelperFunctions import pareto_frontier
from vrtool.flood_defence_system.dike_traject import DikeTraject

class RandomizedParetoFrontier(StrategyBase):
    # Old Pareto Routine: evaluates random combinations of measures.
    def fill_with_MIP(self, mip_results):
        self.LCCOption = mip_results.LCCOption
        self.RiskGeotechnical = mip_results.RiskGeotechnical
        self.RiskOverflow = mip_results.RiskOverflow
        self.opt_parameters = mip_results.opt_parameters
        TC = np.empty(
            (
                self.opt_parameters["N"],
                self.opt_parameters["Sh"],
                self.opt_parameters["Sg"],
            )
        )
        for i in range(0, self.opt_parameters["N"]):
            for ij in range(0, self.opt_parameters["Sh"]):
                for ijk in range(0, self.opt_parameters["Sg"]):
                    TC[i, ij, ijk] = (
                        self.LCCOption[i, ij, ijk]
                        + np.sum(self.RiskGeotechnical[i, ijk, :])
                        + np.sum(self.RiskOverflow[i, ij, :])
                    )
        self.TC = TC

    def evaluate(
        self,
        traject: DikeTraject,
        solutions: Dict[str, Solutions],
        output_path: Path,
        splitparams=False,
        NrSets=1,
        NrSamples=100,
        greedystrategy=False,
        StartSet=0,
    ):
        """
        TODO: This method does not follow the parent implementation.
        """
        self.option_combis = []
        self.LCC_combis = []
        self.TotalRisk_combis = []
        # filtering
        # for each section
        LCC_ind = np.empty(np.shape(self.LCCOption), dtype=np.int32)
        LCC_ind[:, :, :] = np.argsort(self.LCCOption[:, :, :], axis=2)
        TC_sorted = copy.deepcopy(self.TC)
        LCC_sorted = copy.deepcopy(self.LCCOption)

        for n in range(0, self.opt_parameters["N"]):
            for sh in range(0, self.opt_parameters["Sh"]):
                TC_sorted[n, sh, :] = TC_sorted[n, sh, LCC_ind[n, sh, :]]
                LCC_sorted[n, sh, :] = LCC_sorted[n, sh, LCC_ind[n, sh, :]]
                TCmin = TC_sorted[n, sh, 0]
                for sg in range(0, self.opt_parameters["Sg"]):
                    if TC_sorted[n, sh, sg] > TCmin:
                        if self.TC[n, sh, LCC_ind[n, sh, sg]] < 1e90:
                            self.TC[n, sh, LCC_ind[n, sh, sg]] = 1e99

        relevant_indices = pd.DataFrame(
            np.argwhere(self.TC < 1e20), columns=["N", "Sh", "Sg"]
        )
        if greedystrategy:
            set_range = NrSets + 1
        else:
            set_range = NrSets
        for j in range(StartSet, set_range):
            option_sizes = []
            for i in range(0, self.opt_parameters["N"]):
                option_sizes.append(
                    len(relevant_indices.loc[relevant_indices["N"] == i])
                )

            if greedystrategy and j == set_range - 1:
                measures = copy.deepcopy(
                    greedystrategy.TakenMeasures.sort_values("Section")
                )
                option_index_list = np.empty(
                    (len(greedystrategy.TakenMeasures.sort_values("Section")) - 1, 3)
                )  # N, Sh, Sg
                n_list = []
                sh_list = []
                sg_list = []
                sec_count = -1
                for i in measures.Section.unique():
                    if "DV" in i:
                        sec_count += 1
                        for ind, jj in measures.loc[
                            measures["Section"] == i
                        ].iterrows():
                            # find height index
                            # split the string for height
                            id = jj["ID"].split("+")
                            if id[0] in greedystrategy.options_height[i]["ID"].values:
                                # find the index and put it in the optionslist
                                filtered = (
                                    greedystrategy.options_height[i]
                                    .loc[
                                        greedystrategy.options_height[i]["ID"] == id[0]
                                    ]
                                    .loc[
                                        greedystrategy.options_height[i]["dcrest"]
                                        == jj["dcrest"]
                                    ]
                                )
                                if filtered.shape[0] == 1:
                                    sh_list.append(filtered.index[0])
                                elif filtered.shape[0] == 0:
                                    sh_list.append("leeggefilterd")
                                else:
                                    raise ValueError(
                                        "multiple records found after filtering"
                                    )
                            elif len(id) > 1:
                                if (
                                    id[1]
                                    in greedystrategy.options_height[i]["ID"].values
                                ):
                                    # find the index and put it in the optionslist
                                    filtered = (
                                        greedystrategy.options_height[i]
                                        .loc[
                                            greedystrategy.options_height[i]["ID"]
                                            == id[1]
                                        ]
                                        .loc[
                                            greedystrategy.options_height[i]["dcrest"]
                                            == jj["dcrest"]
                                        ]
                                    )
                                    if filtered.shape[0] == 1:
                                        sh_list.append(filtered.index[0])
                                    elif filtered.shape[0] == 0:
                                        sh_list.append(0)
                                    else:
                                        raise ValueError(
                                            "multiple records found after filtering"
                                        )
                            else:
                                sh_list.append(0)

                            if (
                                jj["ID"]
                                in greedystrategy.options_geotechnical[i]["ID"].values
                            ):
                                # find index an put in optionslist
                                filtered = (
                                    greedystrategy.options_geotechnical[i]
                                    .loc[
                                        greedystrategy.options_geotechnical[i]["ID"]
                                        == jj["ID"]
                                    ]
                                    .loc[
                                        greedystrategy.options_geotechnical[i]["dberm"]
                                        == jj["dberm"]
                                    ]
                                    .loc[
                                        greedystrategy.options_geotechnical[i]["yes/no"]
                                        == jj["yes/no"]
                                    ]
                                )
                                if filtered.shape[0] == 1:
                                    sg_list.append(filtered.index[0])
                                    n_list.append(sec_count)
                                elif filtered.shape[0] > 1:
                                    filtered = filtered.loc[
                                        filtered["dcrest"] == jj["dcrest"]
                                    ]
                                    if filtered.shape[0] == 1:
                                        sg_list.append(filtered.index[0])
                                        n_list.append(sec_count)
                                    else:
                                        raise ValueError("no idea what to do")
                                elif filtered.shape[0] == 0:
                                    sg_list.append(0)

                                else:
                                    raise ValueError(
                                        "multiple records found after filtering"
                                    )
                combinations = np.array([n_list, sh_list, sg_list]).T
                # correct for do nothing:
                combinations[:, 1:] += 1
                # combinations[np.argwhere(combinations[:,1]==1),1] -= 1
                # combinations[np.argwhere(combinations[:,2]==1),2] -= 1

                # do nothings
                combinations_extra = np.zeros(
                    (self.opt_parameters["N"], 3), dtype=np.int32
                )
                combinations_extra[:, 0] = np.arange(
                    0, self.opt_parameters["N"], 1, dtype=np.int32
                )
                combinations = np.concatenate((combinations, combinations_extra))
                sorted_combi = []
                for i in range(0, len(np.unique(combinations[:, 0]))):
                    sorted_combi.append(np.array([]))
                for i in np.unique(combinations[:, 0]):
                    sorted_combi[i] = combinations[
                        np.argwhere(combinations[:, 0] == i), :
                    ]

                # translate to option_combis
                option_combis = list(itertools.product(*sorted_combi))
                option_combis = np.array(option_combis).reshape(
                    np.array(option_combis).shape[0],
                    np.array(option_combis).shape[1],
                    np.array(option_combis).shape[3],
                )
            # only use all combinations if the total number is < 1e6
            elif np.product(option_sizes) < 1e1:
                option_sizes = []

                for i in self.options.keys():
                    option_sizes.append(range(0, np.size(self.options[i], 0)))
                option_combis = list(itertools.product(*option_sizes))
            else:
                # draw N samples for each section in the range 0 to option size
                # then construct an array with all the indices
                option_combis = np.empty(
                    (NrSamples, self.opt_parameters["N"], 3), dtype=np.int32
                )
                for i in range(0, self.opt_parameters["N"]):
                    # what are the relevant indices?
                    rel_ind = np.array(relevant_indices.loc[relevant_indices["N"] == i])
                    # sample integers of range
                    indices = np.random.randint(0, len(rel_ind), (NrSamples,))

                    # write indices
                    option_combis[:, i, :] = rel_ind[indices, :]

            LCC = np.zeros((option_combis.shape[0],))
            RiskOverflow = np.zeros(
                (option_combis.shape[0], self.RiskOverflow.shape[2])
            )
            RiskGeotechnical = np.zeros((option_combis.shape[0],))

            for i in range(0, option_combis.shape[0]):
                for n in range(0, self.opt_parameters["N"]):
                    if (
                        self.LCCOption[
                            option_combis[i, n, 0],
                            option_combis[i, n, 1],
                            option_combis[i, n, 2],
                        ]
                        > 1e90
                    ):
                        if option_combis[i, n, 1] == 0:
                            if (
                                not self.LCCOption[
                                    option_combis[i, n, 0], 1, option_combis[i, n, 2]
                                ]
                                > 1e90
                            ):
                                option_combis[i, n, 1] = 1
                        elif option_combis[i, n, 1] == 1:
                            if (
                                not self.LCCOption[
                                    option_combis[i, n, 0], 0, option_combis[i, n, 2]
                                ]
                                > 1e90
                            ):
                                option_combis[i, n, 1] = 0

                    LCC[i] += self.LCCOption[
                        option_combis[i, n, 0],
                        option_combis[i, n, 1],
                        option_combis[i, n, 2],
                    ]
                    RiskOverflow[i, :] = np.max(
                        np.array(
                            [
                                RiskOverflow[i, :],
                                self.RiskOverflow[
                                    option_combis[i, n, 0], option_combis[i, n, 1]
                                ],
                            ]
                        ),
                        axis=0,
                    )
                    RiskGeotechnical[i] += np.sum(
                        self.RiskGeotechnical[
                            option_combis[i, n, 0], option_combis[i, n, 2]
                        ]
                    )

            RiskOverflow_summed = np.sum(RiskOverflow, axis=1)
            self.option_combis.append(option_combis)
            self.LCC_combis.append(LCC)
            self.TotalRisk_combis.append(np.add(RiskOverflow_summed, RiskGeotechnical))
            Results = pd.DataFrame(
                np.array(
                    [
                        LCC,
                        np.add(RiskOverflow_summed, RiskGeotechnical),
                        np.add(LCC, np.add(RiskOverflow_summed, RiskGeotechnical)),
                    ]
                ).T,
                columns=["LCC", "TR", "TC"],
            )
            if greedystrategy and j == set_range - 1:
                Results.to_csv(output_path.joinpath("ParetoResultsGreedy.csv"))
                print("Set " + str(j + 1) + " of " + str(set_range) + " finished")
            else:
                p_frontX, p_frontY, index = pareto_frontier(
                    Xs=Results["LCC"].values,
                    Ys=Results["TR"].values,
                    maxX=False,
                    maxY=False,
                )
                Results.iloc[index].to_csv(
                    output_path.joinpath("ParetoResults" + str(j) + ".csv")
                )
                print("Set " + str(j + 1) + " of " + str(set_range) + " finished")

