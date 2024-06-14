from collections.abc import Callable

from tests.orm import with_empty_db_context
from vrtool.orm.models import AssessmentMechanismResult
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.orm_base_model import OrmBaseModel


class TestAssessmentMechanismResult:
    @with_empty_db_context
    def test_initialize_with_database_fixture(
        self, get_basic_mechanism_per_section: Callable[[], MechanismPerSection]
    ):
        # 1. Define test data.
        _mechanism_per_section = get_basic_mechanism_per_section()

        # 2. Run test
        _assessment_mechanism_result = AssessmentMechanismResult.create(
            beta=3.1234,
            time=0.0,
            mechanism_per_section=_mechanism_per_section,
        )

        # 3. Verify expectations.
        assert isinstance(_assessment_mechanism_result, AssessmentMechanismResult)
        assert isinstance(_assessment_mechanism_result, OrmBaseModel)
        assert (
            _assessment_mechanism_result.mechanism_per_section == _mechanism_per_section
        )
        assert (
            _assessment_mechanism_result
            in _mechanism_per_section.assessment_mechanism_results
        )
