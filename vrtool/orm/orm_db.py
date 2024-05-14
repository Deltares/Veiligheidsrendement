from peewee import SqliteDatabase

# Set pragma "foreign_keys" to True to force cascade on delete.
vrtool_db = SqliteDatabase(None, pragmas={"foreign_keys": 1})
