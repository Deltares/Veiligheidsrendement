import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from vrtool.common.enums.mechanism_enum import MechanismEnum


class MechanismInput:
    # Class for input of a mechanism
    def __init__(self, mechanism: MechanismEnum):
        self.mechanism = mechanism
        self.input = {}
