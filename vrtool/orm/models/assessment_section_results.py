from peewee import ForeignKeyField, FloatField

from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name
from vrtool.orm.models.section_data import SectionData


class AssessmentSectionResults(OrmBaseModel):
    beta = FloatField()
    time = FloatField()
    section_id = ForeignKeyField(SectionData, backref="assessment_results")

    class Meta:
        table_name = _get_table_name(__qualname__)
