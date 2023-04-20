try:
    import cPickle as pickle
except:
    import pickle

import collections
import copy
from pathlib import Path
from shutil import copyfile, rmtree

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import pandas as pd
from openpyxl import load_workbook
from pandas import DataFrame

from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject

"""
TODO: [CS] Having such file is dangerous as logic should be placed where needed and proper relationships / inheritance used where needed.
    - Readers and writers should ideally be placed under a io module instead of here.
    - Calculations should be framed within the used modules.
    - Collection(s) methods should also be localized, and in case they are ALWAYS used then such object should be reconsidered instead.
    - Postprocessing functions should be placed under the (new) `post_processing` module.
"""


def ld_write_object(file_path: Path, object_to_write):
    """
    Write to file with cPickle/pickle (as binary)
    """
    with file_path.open("wb") as _pickle_file:
        _pickle_file.write(pickle.dumps(object_to_write, 1))


def ld_read_object(file_path: Path):
    """
    Used to read a pickle file which can objects.
    This script can load the pickle file so you have a nice object (class or dictionary)
    """
    _loaded_data = None
    with file_path.open("rb") as _pickle_file:
        _loaded_data = pickle.load(_pickle_file)
    return _loaded_data


def flatten_dictionary(nested_dictionary, parent_key: str = "", sep: str = "_") -> dict:
    """
    Helper function to flatten a nested dictionary
    """
    items = []
    for k, v in nested_dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dictionary(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def draw_alpha_bar_plot(
    result_list: list,
    xlabels=None,
    Pvalues=None,
    method="MCS",
    suppress_ind=None,
    title=None,
):
    """
    Draw stacked bars for the importance factors of a list of openturns FORM and results
    TODO: This method would probably fail as `getImportanceFactors` can't be found across the current codebase.
    """
    idx = np.nonzero(Pvalues)[0][0]
    if method == "MCS":
        labels = result_list[idx].getImportanceFactors().getDescription()
    elif method == "FORM":
        labels = (
            result_list[idx]
            .getImportanceFactors(ot.AnalyticalResult.CLASSICAL)
            .getDescription()
        )

    alphas = []
    for i in range(idx, len(result_list)):
        alpha = (
            list(result_list[i].getImportanceFactors(ot.AnalyticalResult.CLASSICAL))
            if method == "FORM"
            else list(result_list[i].getImportanceFactors())
        )
        if suppress_ind != None:
            for ix in suppress_ind:
                alpha[ix] = 0.0
        alpha = np.array(alpha) / np.sum(alpha)
        alphas.append(alpha)

    alfas = DataFrame(alphas, columns=labels)
    alfas.plot.bar(stacked=True, label=xlabels)

    if Pvalues != None:
        plt.plot(range(0, len(xlabels)), Pvalues, "b", label="Fragility Curve")

    xlabels = ["{:4.2f}".format(xlabels[i]) for i in range(0, len(xlabels))]
    plt.xticks(range(0, len(xlabels)), xlabels)
    plt.legend()
    plt.title(title)
    plt.show()


def calculate_r_exit(
    h_exit: float, k: float, d_cover, D, wl: float, Lbase, Lachter, Lfore=0
) -> float:
    """
    k = 0.0001736111
    h_exit = 2.5
    wl = 6.489
    Lbase = 36
    Lachter = 5.65
    """

    lambda2 = np.sqrt(((k * 86400) * D) * (d_cover / 0.01))
    # slight modification: foreshore is counted as dijkzate
    phi2 = h_exit + (wl - h_exit) * (
        (lambda2 * np.tanh(2000 / lambda2))
        / (
            lambda2 * np.tanh(Lfore / lambda2)
            + Lbase
            + Lachter
            + lambda2 * np.tanh(2000 / lambda2)
        )
    )
    r_exit = (phi2 - h_exit) / (wl - h_exit)
    return r_exit


def adapt_input(
    grid_data: pd.DataFrame, monitored_sections: DikeSection, base_case: DikeTraject
) -> list[DikeTraject]:
    _geo_risk_cases = []
    for i, row in grid_data.iterrows():
        _geo_risk_cases.append(copy.deepcopy(base_case))
        # adapt k-value
        for j in _geo_risk_cases[-1].sections:
            if j.name in monitored_sections:
                for ij in list(j.Reliability.Mechanisms["Piping"].Reliability.keys()):
                    j.Reliability.Mechanisms["Piping"].Reliability[ij].Input.input[
                        "k"
                    ] = row["k"]
                    wl = np.array(
                        j.Reliability.Load.distribution.computeQuantile(
                            1 - _geo_risk_cases[-1].GeneralInfo["Pmax"]
                        )
                    )[0]
                    new_r_exit = calculate_r_exit(
                        j.Reliability.Mechanisms["Piping"]
                        .Reliability[ij]
                        .Input.input["h_exit"],
                        j.Reliability.Mechanisms["Piping"]
                        .Reliability[ij]
                        .Input.input["k"],
                        j.Reliability.Mechanisms["Piping"]
                        .Reliability[ij]
                        .Input.input["d_cover"],
                        j.Reliability.Mechanisms["Piping"]
                        .Reliability[ij]
                        .Input.input["D"],
                        wl,
                        j.Reliability.Mechanisms["Piping"]
                        .Reliability[ij]
                        .Input.input["Lvoor"],
                        j.Reliability.Mechanisms["Piping"]
                        .Reliability[ij]
                        .Input.input["Lachter"],
                    )
                    j.Reliability.Mechanisms["Piping"].Reliability[ij].Input.input[
                        "r_exit"
                    ] = new_r_exit

            _geo_risk_cases[-1].GeneralInfo["P_scen"] = row["p"]
    return _geo_risk_cases


def replace_names(
    strategy: StrategyBase, solutions_list: list[Solutions]
) -> StrategyBase:
    strategy.TakenMeasures = strategy.TakenMeasures.reset_index(drop=True)
    for i in range(1, len(strategy.TakenMeasures)):
        # names = TestCaseStrategy.TakenMeasures.iloc[i]['name']
        #
        # #change: based on ID and get Names from new table.
        # if isinstance(names, list):
        #     for j in range(0, len(names)):
        #         names[j] = TestCaseSolutions[TestCaseStrategy.TakenMeasures.iloc[i]['Section']].Measures[names[j]].parameters['Name']
        # else:
        #     names = TestCaseSolutions[TestCaseStrategy.TakenMeasures.iloc[i]['Section']].Measures[names].parameters['Name']
        id = strategy.TakenMeasures.iloc[i]["ID"]
        if isinstance(id, list):
            id = "+".join(id)

        section = strategy.TakenMeasures.iloc[i]["Section"]
        name = (
            solutions_list[section]
            .measure_table.loc[solutions_list[section].measure_table["ID"] == id][
                "Name"
            ]
            .values
        )
        strategy.TakenMeasures.at[i, "name"] = name
    return strategy


def read_batch_info(input_dir: Path) -> list[str]:
    """
    A more generic function to read and write data from and to a shelve. But it is not implemented fully
    """
    _case_set = pd.read_csv(input_dir / "CaseSet.csv", delimiter=";")
    _measure_sets = pd.read_csv(input_dir / "MeasureSet.csv", delimiter=";")
    _available_sections = [
        p for p in input_dir.joinpath("BaseData").iterdir() if p.is_file()
    ]

    _case_list = []
    for row in _case_set.iterrows():
        if row[1]["CaseOn"] == 1:

            _case_no = row[1]["CaseNumber"]
            _case_measure_set = pd.DataFrame(
                {"available": _measure_sets[row[1]["MeasureSet"]].values}
            )
            for run in range(1, row[1]["Runs"] + 1):
                _sections = np.random.randint(
                    0, len(_available_sections), size=row[1]["Sections"]
                )

                # make directory
                _case_path = input_dir.joinpath(
                    "{:02d}".format(_case_no), "{:03d}".format(run)
                )
                # if the case exists: delete the entire directory
                if _case_path.exists():
                    rmtree(_case_path)
                _case_list.append(("{:02d}".format(_case_no), "{:03d}".format(run)))
                _case_path.joinpath("StabilityInner").mkdir(parents=True, exist_ok=True)
                _case_path.joinpath("Piping").mkdir(parents=True, exist_ok=True)
                _case_path.joinpath("Overflow").mkdir(parents=True, exist_ok=True)
                _case_path.joinpath("Toetspeil").mkdir(parents=True, exist_ok=True)
                # copy the section files
                for i in _sections:
                    copyfile(
                        _available_sections[i],
                        _case_path.joinpath(_available_sections[i].name),
                    )
                    book = load_workbook(
                        _case_path.joinpath(_available_sections[i].name)
                    )
                    writer = pd.ExcelWriter(
                        _case_path.joinpath(_available_sections[i].name),
                        engine="openpyxl",
                    )
                    writer.book = book
                    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                    _case_measure_set.to_excel(
                        writer,
                        "Measures",
                        columns=["available"],
                        index=False,
                        startcol=3,
                    )
                    writer.save()
                    # change the measureset
                # then copy all the needed files for
                # StabilityInner
                for p in _case_path.iterdir():
                    if p.is_file():
                        data = pd.read_excel(p, sheet_name="General", index_col=0)
                        _base_data_dir = input_dir / "BaseData"
                        copyfile(
                            _base_data_dir.joinpath(data.loc["Overflow"]["Value"]),
                            _case_path.joinpath(data.loc["Overflow"]["Value"]),
                        )
                        copyfile(
                            _base_data_dir.joinpath(
                                data.loc["StabilityInner"]["Value"]
                            ),
                            _case_path.joinpath(data.loc["StabilityInner"]["Value"]),
                        )
                        copyfile(
                            _base_data_dir.joinpath(data.loc["Piping"]["Value"]),
                            _case_path.joinpath(data.loc["Piping"]["Value"]),
                        )
                        copyfile(
                            _base_data_dir.joinpath(
                                "Toetspeil", data.loc["LoadData"]["Value"]
                            ),
                            _case_path.joinpath(
                                "Toetspeil", data.loc["LoadData"]["Value"]
                            ),
                        )
    return _case_list


def calc_r_squared(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    SStot = np.sum(np.subtract(x, np.mean(x)) ** 2)
    SSreg = np.sum(np.subtract(y, np.mean(x)) ** 2)
    SSres = np.sum(np.subtract(x, y) ** 2)
    Rsq = 1 - np.divide(SSres, SStot)
    return Rsq
