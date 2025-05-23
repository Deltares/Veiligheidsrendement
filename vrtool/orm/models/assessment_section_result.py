from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name
from vrtool.orm.models.section_data import SectionData


class AssessmentSectionResult(OrmBaseModel):
    beta = FloatField()
    time = IntegerField()
    section_data = ForeignKeyField(
        SectionData, backref="assessment_section_results", on_delete="CASCADE"
    )

    class Meta:
        table_name = _get_table_name(__qualname__)
