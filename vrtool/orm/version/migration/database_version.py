from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import peewee

from vrtool.orm.models.version import Version as DbVersion
from vrtool.orm.orm_controllers import open_database
from vrtool.orm.version.orm_version import OrmVersion


@dataclass
class DatabaseVersion(OrmVersion):
    """
    Class representing a versioned database.
    """

    database_path: Path

    @classmethod
    def from_database(cls, database_path: Path) -> DatabaseVersion:
        """
        Create a DatabaseVersion object from the database at the given path.

        Args:
            database_path (Path): Path to the database file.

        Returns:
            DatabaseVersion: Object representing a versioned database.
        """

        def parse_version(version_string: str) -> tuple[int, int, int]:
            return tuple(map(int, version_string.split(".")))

        with open_database(database_path).connection_context():
            _db_version_str = "0.1.0"
            try:
                _db_version_str = DbVersion.get().orm_version
            except peewee.OperationalError as _op_err:
                logging.error(_op_err)
        _major, _minor, _patch = parse_version(_db_version_str)
        return cls(
            major=_major, minor=_minor, patch=_patch, database_path=database_path
        )
