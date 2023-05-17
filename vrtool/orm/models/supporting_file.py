from peewee import CharField, ForeignKeyField

from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
    _max_char_length,
)


class SupportingFile(OrmBaseModel):
    computation_scenario = ForeignKeyField(
        ComputationScenario, backref="supporting_files"
    )
    filename = CharField(max_length=_max_char_length)

    class Meta:
        table_name = _get_table_name(__qualname__)
