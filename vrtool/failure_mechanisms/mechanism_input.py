from dataclasses import dataclass, field
from typing import Any

from vrtool.common.enums.mechanism_enum import MechanismEnum


@dataclass
class MechanismInput:
    """
    This class is used to store the inputs for a failure mechanism.
    """

    mechanism: MechanismEnum
    input: dict[str, Any] = field(default_factory=dict)
