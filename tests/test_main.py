import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import CliRunner

from tests import test_data, test_results
from vrtool import __main__
from vrtool.common.enums.mechanism_enum import MechanismEnum


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

    def test_given_valid_path_without_config_when_run_full_then_fails(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _input_dir = test_results / request.node.name
        if not _input_dir.exists():
            _input_dir.mkdir(parents=True)

        assert _input_dir.exists()

        # 2. Run test.
        _run_result = CliRunner().invoke(
            __main__.run_full,
            [str(_input_dir)],
        )

        # 3. Verify expectations.
        assert _run_result.exit_code == 1

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
        json_file = _input_dir.joinpath("test_config.json")
        json_file.touch()
        json_file.write_text(json.dumps(json_config, indent=4))

        yield _input_dir, _output_dir

        json_file.unlink()
        _test_db_file.unlink()

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "cli_config_dirs",
        ["38-1 two river sections"],
        indirect=True,
    )
    def test_given_valid_input_when_run_full_then_succeeds(
        self, cli_config_dirs: tuple[Path, Path]
    ):
        # NOTE: Keep the test case as the fastest of all
        # available in `api_acceptance_cases`.
        # 1. Define test data.
        _input_dir, _output_dir = cli_config_dirs
        assert _input_dir.exists()
        assert not _output_dir.exists()

        # 2. Run test.
        _run_result = CliRunner().invoke(
            __main__.run_full,
            [str(_input_dir)],
        )

        # 3. Verify final expectations.
        assert _run_result.exit_code == 0
        assert _output_dir.exists()
        assert any(_output_dir.glob("*"))
