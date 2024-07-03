from pathlib import Path

from vrtool.orm.version.increment_type_enum import IncrementTypeEnum


class OrmVersion:
    major: int
    minor: int
    patch: int

    def __init__(self, version_file: Path | None) -> None:
        if version_file:
            self.version_file = version_file
        else:
            self.version_file = Path(__file__).parent.joinpath("__init__.py")
        (self.major, self.minor, self.patch) = self.read_version()

    @property
    def version_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def version_dict(self) -> dict[IncrementTypeEnum, int]:
        return {
            IncrementTypeEnum.MAJOR: self.major,
            IncrementTypeEnum.MINOR: self.minor,
            IncrementTypeEnum.PATCH: self.patch,
        }

    def read_version(self) -> tuple[int, int, int]:
        with open(self.version_file, "r") as f:
            for line in f:
                if "__version__" in line:
                    _version = line.split('"')[1]
                    return tuple(map(int, _version.split(".")))
        return (0, 0, 0)

    def update_version(self, new_version: tuple[int, int, int]) -> None:
        self.major = new_version[0]
        self.minor = new_version[1]
        self.patch = new_version[2]
        with open(self.version_file, "w") as f:
            f.write(f'__version__ = "{self.version_string}"\n')

    def add_increment(self, increment_type: IncrementTypeEnum):
        _version = self.version_dict
        _version[increment_type] += 1
        self.update_version(tuple(_version.values()))
