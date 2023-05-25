from peewee import ForeignKeyField

from vrtool.orm.models.measure import Measure
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name
from vrtool.orm.models.section_data import SectionData


class MeasurePerSection(OrmBaseModel):
    section = ForeignKeyField(SectionData, backref="measures_per_section")
    measure = ForeignKeyField(Measure, backref="sections_per_measure")

    class Meta:
        table_name = _get_table_name(__qualname__)
