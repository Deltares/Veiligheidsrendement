import pytest

with_empty_db_fixture = pytest.mark.usefixtures("empty_db_context")
with_custom_measures_vrtool_config = pytest.mark.usefixtures(
    "custom_measures_vrtool_config"
)
