from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from vrtool.orm.models.version import Version as DbVersion
from vrtool.orm.orm_db import vrtool_db
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
        A direct connection to the database is made to retrieve the version
        to avoid dependency on the ORM definition.

        Args:
            database_path (Path): Path to the database file.

        Returns:
            DatabaseVersion: Object representing a versioned database.
        """

        def parse_version(version_string: str) -> tuple[int, int, int]:
            return tuple(map(int, version_string.split(".")))

        with sqlite3.connect(database_path) as _db_connection:
            _db_version_str = "0.1.0"  # default version
            _query = "SELECT orm_version FROM Version;"
            try:
                _db_version = _db_connection.execute(_query).fetchone()
                if _db_version:
                    _db_version_str = _db_version[0]
            except sqlite3.OperationalError as _op_err:
                if str(_op_err) == "no such table: Version":
                    logging.info(
                        "Database versie tabel niet gevonden. Versie 0.1.0 wordt aangenomen."
                    )
                else:
                    raise RuntimeError("Error reading database version") from _op_err

        _major, _minor, _patch = parse_version(_db_version_str)
        return cls(
            major=_major, minor=_minor, patch=_patch, database_path=database_path
        )

    def update_version(self, version: OrmVersion) -> None:
        """
        Update the database version to the given version.

        Args:
            database_path (Path): Path to the database file.
            version (OrmVersion): Version to update the database to.
        """
        self.major = version.major
        self.minor = version.minor
        self.patch = version.patch

        # Don't use the OrmControllers to avoid circular dependencies.
        vrtool_db.init(self.database_path)
        vrtool_db.connect()
        with vrtool_db.connection_context():
            _version = DbVersion.get_or_none()
            if not _version:
                _version = DbVersion()
            _version.orm_version = str(version)
            _version.save()
