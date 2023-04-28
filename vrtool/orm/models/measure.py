from vrtool.orm.models.base_model import BaseModel, _get_table_name, _max_char_length
from peewee import ForeignKeyField, CharField, IntegerField
from vrtool.orm.models.measure_type import MeasureType
from vrtool.orm.models.combinable_type import CombinableType

class Measure(BaseModel):
    """
    This should be defined as an abstract class.
    """
    measure_type = ForeignKeyField(MeasureType, backref="measures")
    combinable_type = ForeignKeyField(CombinableType, backref="measures")
    name = CharField(max_length=_max_char_length)
    year = IntegerField(default=2025)

    class Meta:
        table_name = _get_table_name(__qualname__)
