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
        self.read_version()

    @property
    def version_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def read_version(self) -> tuple[int, int, int]:
        _version = (0, 0, 0)
        if self.version_file.exists():
            with open(self.version_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "__version__" in line:
                        _version_str = line.split('"')[1]
                        _version = tuple(map(int, _version_str.split(".")))
        self.set_version(_version)
        return _version

    def get_version(self) -> tuple[int, int, int]:
        return self.major, self.minor, self.patch

    def set_version(self, version: tuple[int, int, int]) -> None:
        self.major, self.minor, self.patch = version

    def add_increment(self, increment_type: IncrementTypeEnum):
        if increment_type == IncrementTypeEnum.MAJOR:
            self.major += 1
        elif increment_type == IncrementTypeEnum.MINOR:
            self.minor += 1
        elif increment_type == IncrementTypeEnum.PATCH:
            self.patch += 1

    def write_version(self) -> None:
        if not self.version_file.parent.exists():
            self.version_file.parent.mkdir(parents=True)
        with open(self.version_file, "w", encoding="utf-8") as f:
            f.write(f'__version__ = "{self.version_string}"\n')
