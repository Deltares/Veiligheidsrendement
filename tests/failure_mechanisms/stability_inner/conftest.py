from typing import Iterator

import numpy as np
import pytest

from vrtool.failure_mechanisms.mechanism_input import MechanismInput


@pytest.fixture(name="mechanism_input_fixture")
def get_stability_inner_mechanism_input_fixture() -> Iterator[MechanismInput]:
    mechanism_input = MechanismInput("")
    mechanism_input.input["Pf"] = np.array([0.001], dtype=float)
    mechanism_input.input["beta"] = np.array([0.1], dtype=float)

    yield mechanism_input
