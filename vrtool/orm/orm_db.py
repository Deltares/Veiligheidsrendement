"""
The purpose of this module is to make project-wide available our definition of the
`SqliteDatabase`, this is necessary so that all models in `orm.models` have
their `Meta.database` property linked to this database, which is currently done
through inheritance of the `OrmBaseModel`.
"""

from peewee import SqliteDatabase

# Default values based on official Peewee documentation:
# http://docs.peewee-orm.com/en/latest/peewee/database.html#using-sqlite
# http://docs.peewee-orm.com/en/latest/peewee/database.html#recommended-settings
vrtool_db = SqliteDatabase(
    None,
    pragmas={
        "journal_mode": "wal",
        "cache_size": -1 * 64000,  # 64MB
        "foreign_keys": 1,
        "ignore_check_constraints": 0,
        "synchronous": 0,
    },
)
