from typing import Callable

from tests.orm import with_empty_db_fixture
from vrtool.orm.models import AssessmentSectionResult
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.section_data import SectionData


class TestAssessmentSectionResult:
    @with_empty_db_fixture
    def test_initialize_with_database_fixture(
        self, get_orm_basic_dike_section: Callable[[], SectionData]
    ):
        # 1. Define test data.
        _section = get_orm_basic_dike_section()

        # 2. Run test
        _assessment_section_result = AssessmentSectionResult.create(
            beta=3.1234,
            time=0.0,
            section_data=_section,
        )

        # 3. Verify expectations.
        assert isinstance(_assessment_section_result, AssessmentSectionResult)
        assert isinstance(_assessment_section_result, OrmBaseModel)
        assert _assessment_section_result.section_data == _section
        assert _assessment_section_result in _section.assessment_section_results
