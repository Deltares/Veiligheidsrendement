from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def read_data_from_csv(input_path: Path, reference: Union[int, str]) -> pd.DataFrame:
    """
    Args:
        input_path (str): Path to the dataset folder of the corresponding mechanism.
        reference: (str) A reference to use for the calculation.

    Returns:
        pd.DataFrame: The data from the input csv.
    """
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
    return data
