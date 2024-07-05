import pytest

from tests import get_copy_of_reference_directory, get_vrtool_config_test_copy
from vrtool.api import ApiRunWorkflows


@pytest.mark.slow
class TestApiReportedBugs:
    @pytest.mark.parametrize(
        "directory_name",
        [
            pytest.param(
                "test_stability_multiple_scenarios",
                id="Stability case with multiple scenarios [VRTOOL-340]",
            ),
            pytest.param(
                "test_revetment_step_transition_level",
                id="Revetment case with many transition levels [VRTOOL-330]",
            ),
        ],
    )
    def test_given_case_from_reported_bug_run_all_succeeds(
        self, directory_name: str, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _test_case_dir = get_copy_of_reference_directory(directory_name)

        _vrtool_config = get_vrtool_config_test_copy(
            _test_case_dir.joinpath("config.json"), request.node.name
        )
        assert not any(_vrtool_config.output_directory.glob("*"))

        # 2. Run test.
        ApiRunWorkflows(_vrtool_config).run_all()

        # 3. Verify expectations.
        assert any(_vrtool_config.output_directory.glob("*"))
