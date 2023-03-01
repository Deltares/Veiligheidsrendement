import shutil
from pathlib import Path

from click.testing import CliRunner

from src import __main__
from tests import test_data, test_results


class TestMain:
    def test_given_invalid_directory_when_run_full_then_fails(self):
        # 1. Define test data.
        _invalid_path = "not\\a\\path"
        
        # 2. Run test.
        _run_result = CliRunner().invoke(
            __main__.run_full,
            [str(_invalid_path), "sth"],
        )

        # 3. Verify expectations.
        assert _run_result.exit_code == 2

    def test_given_invalid_traject_when_run_full_then_fails(self):
        # 1. Define test data.
        _casename = "integrated_SAFE_16-3_small"
        _input_dir = test_data / _casename

        assert _input_dir.exists()
        
        # 2. Run test.
        _run_result = CliRunner().invoke(
            __main__.run_full,
            [str(_input_dir), ""],
        )

        # 3. Verify expectations.
        assert _run_result.exit_code == 1

    def test_given_valid_input_when_run_full_then_succeeds(self):
        # TODO: Ideally we want a really small test.
        # 1. Define test data.
        _casename = "integrated_SAFE_16-3_small"
        _traject = "16-3"
        _input_dir = test_data / _casename

        assert _input_dir.exists()

        # Ensure we have a clean results dir.
        _results_dir = test_results / "acceptance"
        if _results_dir.exists():
            shutil.rmtree(_results_dir)
        _results_dir.mkdir(parents=True)

        # 2. Run test.
        _run_result = CliRunner().invoke(
            __main__.run_full,
            [str(_input_dir), _traject],
        )

        # 3. Verify final expectations.
        assert _run_result.exit_code == 0