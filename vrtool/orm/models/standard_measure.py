from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name, _max_char_length
from vrtool.orm.models.measure import Measure
from peewee import ForeignKeyField, IntegerField, FloatField, CharField, BooleanField

class StandardMeasure(OrmBaseModel):
    measure = ForeignKeyField(Measure, backref="standard_measure", unique=True)
    max_inward_reinforcement = IntegerField(default=50)
    max_outward_reinforcement = IntegerField(default=0)
    direction = CharField(default='Inward', max_length=_max_char_length)
    crest_step = FloatField(default=0.5)
    max_crest_increase = FloatField(default=2)
    stability_screen = BooleanField(default=0)
    prob_of_solution_failure = FloatField(default = 1/1000)
    failure_probability_with_solution = FloatField(default=10**-12)
    stability_screen_s_f_increase = FloatField(default=0.2)

    class Meta:
        table_name = _get_table_name(__qualname__)
