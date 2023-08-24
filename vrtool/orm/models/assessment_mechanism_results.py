from peewee import ForeignKeyField, FloatField

from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class AssessmentMechanismResults(OrmBaseModel):
    beta = FloatField()
    time = FloatField()
    mechanism_per_section = ForeignKeyField(
        MechanismPerSection, backref="assessment_mechanism_results"
    )

    class Meta:
        table_name = _get_table_name(__qualname__)
