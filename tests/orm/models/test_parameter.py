from tests.orm.models import empty_db_fixture
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.parameter import Parameter
from vrtool.orm.models.section_data import SectionData


class TestParameter:
    def test_initialize_with_database_fixture(self, empty_db_fixture):
        # 1. Define test data.
        _test_dike_traject = DikeTrajectInfo.create(traject_name="123")
        _test_section = SectionData.create(
            dike_traject=_test_dike_traject,
            section_name="TestSection",
            meas_start=2.4,
            meas_end=4.2,
            section_length=123,
            in_analysis=True,
            crest_height=24,
            annual_crest_decline=42,
        )
        _test_mechanism = Mechanism.create(name="TestMechanism")
        _mech_per_section = MechanismPerSection.create(
            section=_test_section, mechanism=_test_mechanism
        )
        _computation_type = ComputationType.create(name="TestComputation")
        _scenario = ComputationScenario.create(
            mechanism_per_section=_mech_per_section,
            computation_type=_computation_type,
            computation_name="Test Computation",
            scenario_name="test_name",
            scenario_probability=0.42,
            probability_of_failure=0.24,
        )

        # 2. Run test.
        _parameter = Parameter.create(
            computation_scenario=_scenario, parameter="TestParameter", value=4.2
        )

        # 3. Verify expectations
        assert isinstance(_parameter, Parameter)
        assert isinstance(_parameter, OrmBaseModel)
        assert _parameter.computation_scenario == _scenario
        assert _parameter.parameter == "TestParameter"
        assert _parameter.value == 4.2

        assert _parameter in _scenario.parameters

    def test_initialize_two_parameters_with_same_name_with_database_fixture(
        self, empty_db_fixture
    ):
        # 1. Define test data.
        _test_dike_traject = DikeTrajectInfo.create(traject_name="123")
        _test_section = SectionData.create(
            dike_traject=_test_dike_traject,
            section_name="TestSection",
            meas_start=2.4,
            meas_end=4.2,
            section_length=123,
            in_analysis=True,
            crest_height=24,
            annual_crest_decline=42,
        )
        _test_mechanism = Mechanism.create(name="TestMechanism")
        _mech_per_section = MechanismPerSection.create(
            section=_test_section, mechanism=_test_mechanism
        )
        _computation_type = ComputationType.create(name="TestComputation")
        _scenario_a = ComputationScenario.create(
            mechanism_per_section=_mech_per_section,
            computation_type=_computation_type,
            computation_name="Test Computation",
            scenario_name="test_name",
            scenario_probability=0.42,
            probability_of_failure=0.42,
        )
        _scenario_b = ComputationScenario.create(
            mechanism_per_section=_mech_per_section,
            computation_type=_computation_type,
            computation_name="Test Computation",
            scenario_name="test_name",
            scenario_probability=0.24,
            probability_of_failure=0.24,
        )

        # 2. Run test.
        _parameter_a = Parameter.create(
            computation_scenario=_scenario_a, parameter="TestParameter", value=4.2
        )
        _parameter_b = Parameter.create(
            computation_scenario=_scenario_b, parameter="TestParameter", value=2.4
        )

        # 3. Verify expectations
        assert _parameter_a in _scenario_a.parameters
        assert _parameter_b in _scenario_b.parameters
