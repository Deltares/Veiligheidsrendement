from typing import Any, List

import pytest
from typing_extensions import Final

NO_SKIP_OPTION: Final[str] = "--no-skip"


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        NO_SKIP_OPTION,
        action="store_true",
        default=False,
        help="also run skipped tests",
    )


def pytest_collection_modifyitems(config, items: List[Any]):
    """
    Hook to allow overriding of "skipped" tests (NOT CLASSES).
    This is useful to regenerate databases based on tests that
    are usually skipped.

    Reference: https://stackoverflow.com/a/72437511
    """
    if config.getoption(NO_SKIP_OPTION):
        for test in items:
            test.own_markers = [
                marker
                for marker in test.own_markers
                if marker.name not in ("skip", "skipif")
            ]
