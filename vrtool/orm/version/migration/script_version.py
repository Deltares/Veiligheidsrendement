from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vrtool.orm.version.orm_version import OrmVersion


@dataclass
class ScriptVersion(OrmVersion):
    script_path: Path

    def __hash__(self) -> int:
        return (100 * self.major) + (10 * self.minor) + self.patch

    @classmethod
    def from_script(cls, script_path: Path) -> ScriptVersion:
        def parse_version(version_string: str) -> tuple[int, int, int]:
            return tuple(map(int, version_string.replace("v", "").split("_")))

        _major, _minor, _patch = parse_version(script_path.stem)
        return cls(major=_major, minor=_minor, patch=_patch, script_path=script_path)
