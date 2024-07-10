from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name
from vrtool.orm.models.slope_part import SlopePart


class BlockRevetmentRelation(OrmBaseModel):
    slope_part = ForeignKeyField(
        SlopePart, backref="block_revetment_relations", on_delete="CASCADE"
    )
    year = IntegerField()
    top_layer_thickness = FloatField()
    beta = FloatField()

    class Meta:
        table_name = _get_table_name(__qualname__)
