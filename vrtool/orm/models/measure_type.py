from peewee import CharField

from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


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

