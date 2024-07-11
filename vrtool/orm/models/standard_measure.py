from peewee import BooleanField, CharField, FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.measure import Measure
from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
    _max_char_length,
)


class StandardMeasure(OrmBaseModel):
    measure = ForeignKeyField(
        Measure, backref="standard_measure", unique=True, on_delete="CASCADE"
    )
    max_inward_reinforcement = IntegerField(default=50)
    max_outward_reinforcement = IntegerField(default=0)
    direction = CharField(default="Inward", max_length=_max_char_length)
    crest_step = FloatField(default=0.5)
    max_crest_increase = FloatField(default=2)
    stability_screen = BooleanField(default=0)
    prob_of_solution_failure = FloatField(default=1 / 1000)
    max_pf_factor_block = FloatField(default=1000)
    n_steps_block = IntegerField(default=4)

    class Meta:
        table_name = _get_table_name(__qualname__)
