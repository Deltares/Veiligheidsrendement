from peewee import CharField

from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
    _max_char_length,
)


class CharacteristicPointType(OrmBaseModel):
    """
    Possible values:
        * `BIT`,
        * `BUT`,
        * `BUK`,
        * `BIK`,
        * (optionals: `EBL`, `BBL`)
    """

    name = CharField(unique=True, max_length=_max_char_length)

    class Meta:
        table_name = _get_table_name(__qualname__)
