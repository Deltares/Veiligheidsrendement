import pytest

with_empty_db_context = pytest.mark.usefixtures("empty_db_context")
