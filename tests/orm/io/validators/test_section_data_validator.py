import pytest

from vrtool.orm.io.validators.section_data_validator import SectionDataValidator
from vrtool.orm.io.validators.validation_report import ValidationError, ValidationReport
from vrtool.orm.models.section_data import SectionData


class TestSectionDataValidator:
    def test_given_no_section_data_when_validate_then_generates_one_error(self):
        # 1. Given.
        _validator = SectionDataValidator()

        # 2. When.
        _report = _validator.validate(None)

        # 3. Then.
        assert isinstance(_report, ValidationReport)
        assert _report.context is None
        assert len(_report.errors) == 1
        assert _report.errors[0].context_object == None
        assert (
            _report.errors[0].error_message == "No valid value given for SectionData."
        )

    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param(-0.00001, id="Under lower limit"),
            pytest.param(1.0001, id="Over upper limit"),
        ],
    )
    def test_given_invalid_sensitivities_when_validate_then_generates_validation_errors(
        self, invalid_value: float
    ):
        # 1. Given.
        _validator = SectionDataValidator()
        _section_data = SectionData()
        _section_data.sensitive_fraction_piping = invalid_value
        _section_data.sensitive_fraction_stability_inner = invalid_value

        # 2. When.
        _report = _validator.validate(_section_data)

        # 3. Then.
        assert isinstance(_report, ValidationReport)
        assert isinstance(_report.context, SectionData)
        assert len(_report.errors) == 2

        def assert_error(report_error: ValidationError, context_object: str):
            assert report_error.context_object == context_object
            assert (
                report_error.error_message
                == f"'{context_object}' should be a real value in the [0.0, 1.0] limit, but got '{invalid_value}'."
            )

        assert_error(_report.errors[0], "sensitive_fraction_piping")
        assert_error(_report.errors[1], "sensitive_fraction_stability_inner")
