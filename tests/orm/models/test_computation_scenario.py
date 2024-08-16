from typing import Callable

from tests.orm import with_empty_db_context
from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.section_data import SectionData


class TestComputationScenario:
    @with_empty_db_context
    def test_initialize_with_database_fixture(self):
        # 1. Define test data.
        _computation_type = ComputationType.create(name=ComputationTypeEnum.NONE.name)
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
        _test_mech_inst = Mechanism.create(name=MechanismEnum.OVERFLOW.name)
        _mech_per_section = MechanismPerSection.create(
            section=_test_section, mechanism=_test_mech_inst
        )

        # 2. Run test.
        _scenario = ComputationScenario.create(
            mechanism_per_section=_mech_per_section,
            computation_type=_computation_type,
            computation_name="Test Computation",
            scenario_name="test_name",
            scenario_probability=0.42,
            probability_of_failure=0.24,
        )

        # 3. Verify expectations.
        assert isinstance(_scenario, ComputationScenario)
        assert isinstance(_scenario, OrmBaseModel)
        assert not any(_scenario.computation_scenario_parameters)
        assert not any(_scenario.supporting_files)

    @with_empty_db_context
    def test_on_delete_mechanism_per_section_cascades(
        self, get_basic_computation_scenario: Callable[[], ComputationScenario]
    ):
        # 1. Define test data.
        _computation_scenario = get_basic_computation_scenario()
        assert isinstance(_computation_scenario, ComputationScenario)
        assert any(ComputationScenario.select())

        # 2. Run test.
        MechanismPerSection.delete_by_id(
            _computation_scenario.mechanism_per_section.get_id()
        )

        # 3. Verify expectations
        assert not any(ComputationScenario.select())

    @with_empty_db_context
    def test_on_delete_computation_type_cascades(
        self, get_basic_computation_scenario: Callable[[], ComputationScenario]
    ):
        # 1. Define test data.
        _computation_scenario = get_basic_computation_scenario()
        assert isinstance(_computation_scenario, ComputationScenario)
        assert any(ComputationScenario.select())

        # 2. Run test.
        ComputationType.delete_by_id(_computation_scenario.computation_type.get_id())

        # 3. Verify expectations
        assert not any(ComputationScenario.select())
