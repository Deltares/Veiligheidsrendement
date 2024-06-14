import pytest

with_empty_db_fixture = pytest.mark.usefixtures("empty_db_fixture")
with_custom_measures_vrtool_config = pytest.mark.usefixtures(
    "custom_measures_vrtool_config"
)
