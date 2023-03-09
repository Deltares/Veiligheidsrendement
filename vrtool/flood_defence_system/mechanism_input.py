import os
from pathlib import Path

import numpy as np
import pandas as pd

from vrtool.probabilistic_tools.hydra_ring_scripts import read_design_table


class MechanismInput:
    # Class for input of a mechanism
    def __init__(self, mechanism):
        self.mechanism = mechanism
        self.input = {}

    # This routine reads  input from an input sheet
    def fill_mechanism(
        self,
        input_path,
        reference,
        calctype,
        kind="csv",
        sheet=None,
        mechanism=None,
        **kwargs,
    ):
        if kind == "csv":
            if mechanism != "Overflow":
                try:
                    data = pd.read_csv(
                        input_path.joinpath(Path(str(reference)).name),
                        delimiter=",",
                        header=None,
                    )
                except:
                    data = pd.read_csv(
                        input_path.joinpath(Path(str(reference)).name + ".csv"),
                        delimiter=",",
                        header=None,
                    )

                # TODO: fix datatypes in input such that we do not need to drop columns
                data = data.rename(columns={list(data)[0]: "Name"})
                data = data.set_index("Name")
                try:
                    data = data.drop(["InScope", "Opmerking"]).astype(np.float32)
                except:
                    pass

            else:  #'Overflow':
                if calctype == "Simple":
                    data = pd.read_csv(
                        input_path.joinpath(Path(reference).name), delimiter=","
                    )
                    data = data.transpose()
                elif calctype == "HRING":
                    # detect years
                    years = os.listdir(input_path)
                    for count, year in enumerate(years):
                        year_data = read_design_table(
                            input_path.joinpath(year, reference + ".txt")
                        )[["Value", "Beta"]]
                        if count == 0:
                            data = year_data.set_index("Value").rename(
                                columns={"Beta": year}
                            )

                        else:
                            if all(data.index.values == year_data.Value.values):
                                data = pd.concat(
                                    (
                                        data,
                                        year_data.set_index("Value").rename(
                                            columns={"Beta": year}
                                        ),
                                    ),
                                    axis="columns",
                                )
                            # compare value columns:
                        # if count>0 and Value is identical: concatenate.
                        # else: interpolate and then concatenate.
                else:
                    raise Exception("Unknown input type for overflow")

        elif kind == "xlsx":
            data = pd.read_excel(input, sheet_name=sheet)
            data = data.set_index("Name")

        self.temporals = []
        self.char_vals = {}
        for i in range(len(data)):
            # if (data.iloc[i].Name == 'FragilityCurve') and ~np.isnan(data.iloc[i].Value):
            if data.index[i] == "FragilityCurve":
                pass
                # Turned off: old code that doesnt work anymore
            elif calctype == "HRING":
                self.input["hc_beta"] = data
                self.input["h_crest"] = kwargs["crest_height"]
                self.input["d_crest"] = kwargs["dcrest"]
            else:
                x = data.iloc[i][:].values
                if isinstance(x, np.ndarray):
                    if len(x) > 1:
                        self.input[data.index[i]] = x.astype(np.float32)[~np.isnan(x)]
                    elif len(x) == 1:
                        try:
                            if not np.isnan(np.float32(x[0])):
                                self.input[data.index[i]] = np.array([np.float32(x[0])])
                        except:
                            self.input[data.index[i]] = x[0]
                    else:
                        pass
                else:
                    pass

                if data.index[i][-3:] == "(t)":
                    self.temporals.append(data.index[i])

                # for k-value: ensure that value is in m/s not m/d:
                if data.index[i] == "k":
                    try:
                        if any(
                            self.input[data.index[i]] > 1.0
                        ):  # if k>1 it is likely in m/d
                            self.input[data.index[i]] = self.input[data.index[i]] / (
                                24 * 3600
                            )
                            print(
                                "k-value modified as it was likely m/d and should be m/s"
                            )
                    except:
                        if self.input[data.index[i]] > 1.0:
                            self.input[data.index[i]] = self.input[data.index[i]] / (
                                24 * 3600
                            )
                            print(
                                "k-value modified as it was likely m/d and should be m/s"
                            )
