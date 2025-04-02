import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import CliRunner

from tests import get_clean_test_results_dir, test_data, test_results
from vrtool import __main__
from vrtool.common.enums.mechanism_enum import MechanismEnum


class TestMain:
    def test_given_invalid_log_dir_when__initialize_log_file_then_sets_cwd(self):
        # 1. Define test data.
        _invalid_path = None
        _expected_log_dir = Path().cwd()
        # Ensure no log file spresent in the cwd
        def cleanup_cwd_dir():
            for _log_file in _expected_log_dir.glob("*.log"):
                _log_file.unlink()

        cleanup_cwd_dir()
        assert not any(_expected_log_dir.glob("*.log"))

        # 2. Run test.
        __main__._initialize_log_file(_invalid_path)

        # 3. Verify expectations.
        assert any(_expected_log_dir.glob("*.log"))
        cleanup_cwd_dir()

    @pytest.mark.parametrize(
        "time_between_runs, expected_log_files",
        [
            pytest.param(0, 1, id="<60s between runs, 1 log file"),
            pytest.param(61, 2, id=">60s between runs, 2 log files"),
        ],
    )
    def test_given_invalid_directory_when_sequential_runs_within_a_minute_then_creates_one_file(
        self,
        time_between_runs: int,
        expected_log_files: int,
        request: pytest.FixtureRequest,
    ):
        # 1. Define test data.
        _input_dir = get_clean_test_results_dir(request)
        assert _input_dir.exists()

        # 2. Run test.
        for _ in range(2):
            _ = CliRunner().invoke(
                __main__.run_full,
                [str(_input_dir), "--log-dir", str(_input_dir)],
            )
            time.sleep(time_between_runs)

        # 3. Verify expectations.
        assert len(list(_input_dir.glob("vrtool_logging*.log"))) == expected_log_files

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

    def test_given_valid_path_without_config_when_run_full_then_fails(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _input_dir = test_results.joinpath(request.node.name)
        if _input_dir.exists():
            shutil.rmtree(_input_dir)

        _input_dir.mkdir(parents=True)
        assert _input_dir.exists()

        # 2. Run test.
        _run_result = CliRunner().invoke(
            __main__.run_full,
            [str(_input_dir), "--log-dir", str(_input_dir)],
        )

        # 3. Verify expectations.
        assert _run_result.exit_code == 1
        assert any(_input_dir.glob("vrtool_logging*.log"))

    @pytest.fixture(name="cli_config_dirs")
    def _get_cli_config_fixture(
        self, request: pytest.FixtureRequest
    ) -> Iterator[tuple[Path, Path]]:
        _input_dir = test_data.joinpath(request.param)
        _output_dir = test_results.joinpath(request.node.name)
        if _output_dir.exists():
            shutil.rmtree(_output_dir)

        # We need to create a copy of the database on the input directory.
        _reference_db_file = _input_dir.joinpath("vrtool_input.db")
        _test_db_name = "test_{}.db".format(
            hashlib.shake_128(_input_dir.__bytes__()).hexdigest(4)
        )
        _test_db_file = _input_dir.joinpath(_test_db_name)
        if _input_dir.joinpath(_test_db_name).exists():
            _input_dir.joinpath(_test_db_name).unlink()
        shutil.copy(_reference_db_file, _test_db_file)

        json_config = {
            "input_directory": str(_input_dir),
            "input_database_name": _test_db_file.name,
            "traject": "38-1",
            "output_directory": str(_output_dir),
            "excluded_mechanisms": [
                MechanismEnum.REVETMENT.name,
                MechanismEnum.HYDRAULIC_STRUCTURES.name,
            ],
        }

        _json_file = _input_dir.joinpath("test_config.json")
        _json_file.touch()
        _json_file.write_text(json.dumps(json_config, indent=4))

        yield _json_file, _output_dir

        _json_file.unlink()
        _test_db_file.unlink()

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "cli_config_dirs",
        ["38-1_two_river_sections"],
        indirect=True,
    )
    def test_given_valid_input_when_run_full_then_succeeds(
        self, cli_config_dirs: tuple[Path, Path]
    ):
        # NOTE: Keep the test case as the fastest of all
        # available in `api_acceptance_cases`.
        # 1. Define test data.
        _vrtool_config_file, _output_dir = cli_config_dirs
        assert _vrtool_config_file.is_file()
        assert not _output_dir.exists()

        # 2. Run test.
        _run_result = CliRunner().invoke(
            __main__.run_full,
            [str(_vrtool_config_file), "--log-dir", str(_output_dir)],
        )

        # 3. Verify final expectations.
        assert _run_result.exit_code == 0
        assert _output_dir.exists()
        assert any(_output_dir.glob("vrtool_logging*.log"))
