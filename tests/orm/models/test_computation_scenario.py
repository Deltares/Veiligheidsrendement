from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.section_data import SectionData
from tests.orm.models import empty_db_fixture

class TestComputationScenario:

    def test_initialize(self, empty_db_fixture):
        # 1. Define test data.
        _computation_type = ComputationType(name="TestComputation")
        _test_dike_traject = DikeTrajectInfo(traject_name="123")
        _test_section= SectionData(dike_traject=_test_dike_traject, section_name="TestSection")
        _test_mechanism = Mechanism(name="TestMechanism")
        _mech_per_section = MechanismPerSection(section=_test_section, mechanism=_test_mechanism)

        # 2. Run test.
        _scenario = ComputationScenario(mechanism_per_section=_mech_per_section, computation_type = _computation_type, computation_name="Test Computation", scenario_name="test_name", scenario_probability=0.42, probability_of_failure=0.24)

        # 3. Verify expectations.
        assert isinstance(_scenario, ComputationScenario)
        assert isinstance(_scenario, OrmBaseModel)
    
    def test_initialize_without_arguments(self):
        _scenario = ComputationScenario.create()
        assert isinstance(_scenario, ComputationScenario)
        assert isinstance(_scenario, OrmBaseModel)