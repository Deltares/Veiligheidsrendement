from peewee import ForeignKeyField

from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name
from vrtool.orm.models.section_data import SectionData


class MechanismPerSection(OrmBaseModel):
    section = ForeignKeyField(
        SectionData, backref="mechanisms_per_section", on_delete="CASCADE"
    )
    mechanism = ForeignKeyField(
        Mechanism, backref="sections_per_mechanism", on_delete="CASCADE"
    )

    class Meta:
        table_name = _get_table_name(__qualname__)

    @property
    def mechanism_name(self) -> str:
        """
        Retrieves the mechanism's name in capital letters.

        Returns:
            str: The mechanism's name in capital letters.
        """
        return self.mechanism.name.upper()
