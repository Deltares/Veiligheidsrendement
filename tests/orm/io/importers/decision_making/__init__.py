import pytest

with_empty_db_context_and_valid_section_data_without_measures = pytest.mark.usefixtures(
    "empty_db_context", "valid_section_data_without_measures"
)
