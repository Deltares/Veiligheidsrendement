import math

from tests.orm import with_empty_db_context
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.orm_base_model import OrmBaseModel


class TestDikeTrajectInfo:
    @with_empty_db_context
    def test_initialize_with_database_fixture(self):
        # Run test.
        _traject_name = "sth"
        _dike_traject_info = DikeTrajectInfo.create(traject_name=_traject_name)

        # Verify expectations.
        assert isinstance(_dike_traject_info, DikeTrajectInfo)
        assert isinstance(_dike_traject_info, OrmBaseModel)

        # Check default values
        assert _dike_traject_info.traject_name == _traject_name
        assert _dike_traject_info.omega_piping == 0.24
        assert _dike_traject_info.omega_stability_inner == 0.04
        assert _dike_traject_info.omega_overflow == 0.24
        assert math.isnan(_dike_traject_info.a_piping)
        assert _dike_traject_info.b_piping == 300
        assert _dike_traject_info.a_stability_inner == 0.033
        assert _dike_traject_info.b_stability_inner == 50
        assert math.isnan(_dike_traject_info.beta_max)
        assert math.isnan(_dike_traject_info.p_max)
        assert math.isnan(_dike_traject_info.flood_damage)
        assert math.isnan(_dike_traject_info.traject_length)
        assert _dike_traject_info.n_revetment == 3
        assert _dike_traject_info.n_overflow == 1
