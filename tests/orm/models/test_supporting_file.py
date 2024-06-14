from typing import Callable

from tests.orm import with_empty_db_fixture
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.supporting_file import SupportingFile


class TestSupportingFile:
    @with_empty_db_fixture
    def test_initialize_with_database_fixture(
        self, get_basic_computation_scenario: Callable[[], ComputationScenario]
    ):
        # 1. Define test data.
        _scenario = get_basic_computation_scenario()

        # 2. Run test.
        _support_file = SupportingFile.create(
            computation_scenario=_scenario, filename="JustAFile"
        )

        # 3. Verify expectations.
        assert isinstance(_support_file, SupportingFile)
        assert isinstance(_scenario, OrmBaseModel)
        assert _support_file.computation_scenario == _scenario
        assert _support_file in _scenario.supporting_files
