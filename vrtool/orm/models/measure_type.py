from vrtool.orm.models.base_model import OrmBaseModel, _get_table_name
from peewee import CharField

class MeasureType(OrmBaseModel):
    """
    Existing types:
        * Soil reinforcement
        * Stability screen
        * Soil reinforcement with stability screen
        * Vertical Geotextile
        * Diaphragm wall
    """
    name = CharField(unique=True)

    class Meta:
        table_name = _get_table_name(__qualname__)

