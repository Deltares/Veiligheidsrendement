from peewee import CharField, ForeignKeyField

from vrtool.orm.models.combinable_type import CombinableType
from vrtool.orm.models.measure_type import MeasureType
from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
    _max_char_length,
)


class Measure(OrmBaseModel):
    """
    This should be defined as an abstract class.
    """

    measure_type = ForeignKeyField(MeasureType, backref="measures", on_delete="CASCADE")
    combinable_type = ForeignKeyField(
        CombinableType, backref="measures", on_delete="CASCADE"
    )
    name = CharField(max_length=_max_char_length)

    class Meta:
        table_name = _get_table_name(__qualname__)
