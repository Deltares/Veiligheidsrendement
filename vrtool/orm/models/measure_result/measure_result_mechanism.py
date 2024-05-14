from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class MeasureResultMechanism(OrmBaseModel):
    measure_result = ForeignKeyField(
        MeasureResult, backref="measure_result_mechanisms", on_delete="CASCADE"
    )
    mechanism_per_section = ForeignKeyField(
        MechanismPerSection, backref="mechanism_measure_results"
    )

    beta = FloatField()
    time = IntegerField()

    class Meta:
        table_name = _get_table_name(__qualname__)
