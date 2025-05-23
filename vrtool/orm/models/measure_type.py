from peewee import CharField

from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class MeasureType(OrmBaseModel):
    """
    Existing types:
        * Soil reinforcement
        * Stability Screen
        * Soil reinforcement with stability screen
        * Vertical Piping Solution
        * Diaphragm Wall
        * Anchored Sheetpile
    """

    name = CharField(unique=True)

    class Meta:
        table_name = _get_table_name(__qualname__)
