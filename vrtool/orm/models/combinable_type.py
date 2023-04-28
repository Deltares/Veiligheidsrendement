from vrtool.orm.models.base_model import OrmBaseModel, _get_table_name, _max_char_length
from peewee import CharField

class CombinableType(OrmBaseModel):
    """
    Existing types:
        * full
        * combinable
        * partial
    """
    name = CharField(unique=True, max_length=_max_char_length)

    class Meta:
        table_name = _get_table_name(__qualname__)