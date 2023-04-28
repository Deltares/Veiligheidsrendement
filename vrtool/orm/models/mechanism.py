from vrtool.orm.models.orm_base_model import OrmBaseModel, _max_char_length, _get_table_name
from peewee import CharField

class Mechanism(OrmBaseModel):
    name = CharField(max_length=_max_char_length)

    class Meta:
        table_name = _get_table_name(__qualname__)