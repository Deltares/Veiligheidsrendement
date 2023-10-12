import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from vrtool.common.enums import MechanismEnum
from vrtool.failure_mechanisms.mechanism_reader import read_data_from_csv
from vrtool.probabilistic_tools.hydra_ring_scripts import read_design_table


class MechanismInput:
    # Class for input of a mechanism
    def __init__(self, mechanism: MechanismEnum):
        self.mechanism = mechanism
        self.input = {}

    # This routine reads  input from an input sheet
    def fill_mechanism(
        self,
        input_path: Path,
        stix_folder_path: Path,
        externals_path: Path,
        reference: Union[int, str],
        calctype: str,
        mechanism: MechanismEnum = None,
        **kwargs,
    ):
        """
        Args:
            input_path (str): Path to the dataset folder of the corresponding mechanism.
            stix_folder_path (str): Path to the folder containing all the stix files.
            externals_path (str): Path to the folder containing all the externals.
            reference (str): A reference to use for the calculation.
            calctype (str): Calculation type for the given mechanism, one of ['Simple', 'HRING', 'DStability', 'DirectInput'].
            mechanism (MechanismEnum): The mechanism to use for the calculation.
            **kwargs: Additional keyword arguments to pass.

        Returns:
            None.
        """

        if mechanism == MechanismEnum.STABILITY_INNER:
            if calctype == "DStability":
                data = read_data_from_csv(input_path, reference)
                data = data.loc[data.index.isin(["STIXNAAM"])]
                data.loc["STIXNAAM"] = (
                    str(stix_folder_path) + "/" + data.loc["STIXNAAM"]
                )
                data.loc["DStability_exe_path"] = str(externals_path)
            else:
                data = read_data_from_csv(input_path, reference)

        elif mechanism == MechanismEnum.OVERFLOW:
            if calctype == "Simple":
                data = pd.read_csv(
                    input_path.joinpath(Path(reference).name), delimiter=","
                )
                data = data.transpose()
            elif calctype == "HRING":
                # detect years
                for count, year_path in enumerate(input_path.iterdir()):
                    year_data = read_design_table(
                        year_path.joinpath(reference + ".txt")
                    )[["Value", "Beta"]]
                    if count == 0:
                        data = year_data.set_index("Value").rename(
                            columns={"Beta": year_path.stem}
                        )

                    else:
                        if all(data.index.values == year_data.Value.values):
                            data = pd.concat(
                                (
                                    data,
                                    year_data.set_index("Value").rename(
                                        columns={"Beta": year_path.stem}
                                    ),
                                ),
                                axis="columns",
                            )
                        # compare value columns:
                    # if count>0 and Value is identical: concatenate.
                    # else: interpolate and then concatenate.
            else:
                raise Exception("Unknown input type for overflow")

        else:
            data = read_data_from_csv(input_path, reference)

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
                            logging.info(
                                "k-value modified as it was likely m/d and should be m/s"
                            )
                    except:
                        if self.input[data.index[i]] > 1.0:
                            self.input[data.index[i]] = self.input[data.index[i]] / (
                                24 * 3600
                            )
                            logging.info(
                                "k-value modified as it was likely m/d and should be m/s"
                            )
