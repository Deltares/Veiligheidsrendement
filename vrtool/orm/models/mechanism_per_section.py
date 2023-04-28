from vrtool.orm.models.base_model import BaseModel, _get_table_name
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.mechanism import Mechanism
from peewee import ForeignKeyField

class MechanismPerSection(BaseModel):
    section = ForeignKeyField(SectionData, backref="mechanisms_per_section")
    mechanism = ForeignKeyField(Mechanism, backref="sections_per_mechanism")
    class Meta:
        table_name = _get_table_name(__qualname__)