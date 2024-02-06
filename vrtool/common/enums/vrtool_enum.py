from enum import Enum


class VrtoolEnum(Enum):
    def __str__(self) -> str:
        return self.name
