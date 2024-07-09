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
    database_path: Path

    @classmethod
    def from_database(cls, database_path: Path) -> DatabaseVersion:
        with open_database(database_path).connection_context():
            _db_version_str = "0.1.0"
            try:
                _version = DbVersion.get().orm_version
                _db_version_str = _version.get().orm_version
            except peewee.OperationalError as _op_err:
                logging.error(_op_err)
        _major, _minor, _patch = cls.parse_version(_db_version_str)
        return cls(
            major=_major, minor=_minor, patch=_patch, database_path=database_path
        )
