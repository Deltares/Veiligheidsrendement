from tests.orm import empty_db_fixture, get_basic_mechanism_per_section
from vrtool.orm.models import AssessmentMechanismResults
from vrtool.orm.models.orm_base_model import OrmBaseModel


class TestAssessmentMechanismResults:
    def test_initialize_with_database_fixture(self, empty_db_fixture):
        # 1. Define test data.
        _mechanism_per_section = get_basic_mechanism_per_section()

        # 2. Run test
        _assessment_mechanism_results = AssessmentMechanismResults.create(
            beta=3.1234,
            time=0.0,
            mechanism_per_section_id=_mechanism_per_section,
        )

        # 3. Verify expectations.
        assert isinstance(_assessment_mechanism_results, AssessmentMechanismResults)
        assert isinstance(_assessment_mechanism_results, OrmBaseModel)
        assert _assessment_mechanism_results.mechanism_per_section_id == _mechanism_per_section
        assert _assessment_mechanism_results in _mechanism_per_section.assessment_mechanism_results
