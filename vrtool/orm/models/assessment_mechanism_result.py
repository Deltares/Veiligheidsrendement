from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class AssessmentMechanismResult(OrmBaseModel):
    beta = FloatField()
    time = IntegerField()
    mechanism_per_section = ForeignKeyField(
        MechanismPerSection, backref="assessment_mechanism_results", on_delete="CASCADE"
    )

    class Meta:
        table_name = _get_table_name(__qualname__)
