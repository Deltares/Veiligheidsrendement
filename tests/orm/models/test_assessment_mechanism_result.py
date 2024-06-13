import pytest

from tests.orm import get_basic_mechanism_per_section
from vrtool.orm.models import AssessmentMechanismResult
from vrtool.orm.models.orm_base_model import OrmBaseModel


class TestAssessmentMechanismResult:
    @pytest.mark.usefixtures("empty_db_fixture")
    def test_initialize_with_database_fixture(self):
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
