from typing import Iterator

import pytest

from vrtool.defaults.vrtool_config import VrtoolConfig


@pytest.fixture(
    name="invalid_vrtool_config_fixture",
    params=[[0], [100], [0, 100]],
    ids=["Without t=0", "Without t=100", "Without t=0, t=100"],
)
@pytest.mark.parametrize
def get_invalid_vrtool_config_fixture(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[VrtoolConfig, str]]:
    """
    Gets a `VrtoolConfig` instance whose call to `validate` will raise an exception.

    Args:
        request (pytest.FixtureRequest): Inherited argument containing the pytest parameters.

    Yields:
        Iterator[VrtoolConfig]: Invalid `VrtoolConfig` instance.
    """

    def build_invalid_vrtool_config() -> VrtoolConfig:
        _vrtool_config = VrtoolConfig()
        # Exclude values 0 and 100
        _vrtool_config.T = [_t for _t in _vrtool_config.T if _t not in request.param]
        try:
            _vrtool_config.validate_config()
        except ValueError:
            return _vrtool_config
        pytest.fail("This configuration should have thrown a validation error.")

    _expected_error_mssg = (
        "'VrtoolConfig' is niet geldig, het vereist de waarden: 0, 100"
    )

    yield build_invalid_vrtool_config(), _expected_error_mssg
