from typing import Any

from vrtool.common.enums.mechanism_enum import MechanismEnum


class MechanismInput:
    # Class for input of a mechanism
    def __init__(self, mechanism: MechanismEnum):
        self.mechanism = mechanism
        self.input: dict[str, Any] = {}
