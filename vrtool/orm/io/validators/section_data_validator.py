import logging
from math import isclose
from typing import Iterator

from vrtool.orm.io.validators.validation_report import ValidationError, ValidationReport
from vrtool.orm.models.section_data import SectionData


class SectionDataValidator:
    def _valid_sensitivity(self, sensitivity: float) -> bool:
        _max_value = 1.0
        _lower_value = 0.0
        _within_lower_limit = sensitivity > _lower_value or isclose(sensitivity, 0.0, rel_tol=1e-9, abs_tol=1e-9)
        _within_upper_limit = sensitivity < _max_value or isclose(sensitivity, _max_value, rel_tol=1e-9, abs_tol=1e-9)
        return _within_lower_limit and _within_upper_limit

    def _validate_sensitivity(self, section_data: SectionData) -> Iterator[ValidationError]:
        logging.debug("Verifying sensitivy fraction values are in range [0.0, 1.0].")
        def add_sensitivy_error(sensitivity_property: str, value: float):
            return ValidationError(
                f"'{sensitivity_property}' should be a real value in the [0.0, 1.0] limit, but got '{value}'.",
                sensitivity_property)

        if not self._valid_sensitivity(section_data.sensitive_fraction_piping):
            yield add_sensitivy_error("sensitive_fraction_piping", section_data.sensitive_fraction_piping)
        if not self._valid_sensitivity(section_data.sensitive_fraction_stability_inner):
            yield add_sensitivy_error("sensitive_fraction_stability_inner", section_data.sensitive_fraction_stability_inner)

    def validate(self, section_data: SectionData) -> ValidationReport:
        """
        Validates the given `section_data` through its different parameters.

        Args:
            section_data (SectionData): The section data (orm model) to validate.

        Returns:
            ValidationReport: The resulting report with validation results.
        """
        _validation_report = ValidationReport(section_data)

        if not section_data:
            _validation_report.errors.append(
                ValidationError(
                f"No valid value given for {SectionData.__name__}.",
                section_data)
            )
            return _validation_report

        _validation_report.errors.extend(list(self._validate_sensitivity(section_data)))
        return _validation_report



