from peewee import SqliteDatabase

vrtool_db = SqliteDatabase(None, pragmas={"foreign_keys": 1})
