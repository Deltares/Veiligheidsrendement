from peewee import CharField, FloatField, IntegerField

from vrtool.orm.models.orm_base_model import (
    OrmBaseModel,
    _get_table_name,
    _max_char_length,
)


class DikeTrajectInfo(OrmBaseModel):
    traject_name = CharField(max_length=_max_char_length)
    omega_piping = FloatField(default=0.24)
    omega_stability_inner = FloatField(default=0.04)
    omega_overflow = FloatField(default=0.24)
    a_piping = FloatField(default=float("nan"), null=True)
    b_piping = FloatField(default=300)
    a_stability_inner = FloatField(default=0.033)
    b_stability_inner = FloatField(default=50)
    beta_max = FloatField(default=float("nan"), null=True)
    p_max = FloatField(default=float("nan"), null=True)
    flood_damage = FloatField(default=float("nan"), null=True)
    traject_length = FloatField(default=float("nan"), null=True)
    n_revetment = IntegerField(default=3)
    n_overflow = IntegerField(default=1)

    class Meta:
        table_name = _get_table_name(__qualname__)
