from peewee import Model
from vrtool.orm import vrtool_db


class BaseModel(Model):
    class Meta:
        database = vrtool_db