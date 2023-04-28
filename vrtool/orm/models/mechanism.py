from peewee import CharField

from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
    _max_char_length,
)


class Mechanism(OrmBaseModel):
    name = CharField(max_length=_max_char_length)

    class Meta:
        table_name = _get_table_name(__qualname__)