import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests import test_data, test_results
from vrtool import __main__
from vrtool.common.enums import MechanismEnum


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

    @pytest.fixture
    def cli_config_fixture(self, request: pytest.FixtureRequest):
        _test_case = request.param
        _input_dir = test_data / _test_case
        _output_dir = test_results.joinpath(request.node.name)
        if _output_dir.exists():
            shutil.rmtree(_output_dir)

        json_config = {
            "input_directory": str(_input_dir),
            "input_database_name": "vrtool_input.db",
            "traject": "38-1",
            "output_directory": str(_output_dir),
            "excluded_mechanisms": [
                MechanismEnum.REVETMENT.name,
                "HYDRAULIC_STRUCTURES",
            ],
        }
        json_file = _input_dir.joinpath("test_config.json")
        json_file.touch()
        json_file.write_text(json.dumps(json_config, indent=4))

        yield _input_dir, _output_dir

        json_file.unlink()

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "cli_config_fixture",
        ["TestCase1_38-1_no_housing"],
        indirect=True,
    )
    def test_given_valid_input_when_run_full_then_succeeds(
        self, cli_config_fixture: tuple[Path, Path]
    ):
        # TODO: Ideally we want a really small test.
        # 1. Define test data.
        _input_dir, _output_dir = cli_config_fixture
        assert _input_dir.exists()
        assert not _output_dir.exists()

        # Ensure we have a clean results dir.
        _results_dir = _input_dir / "results"
        if _results_dir.exists():
            shutil.rmtree(_results_dir)

        # 2. Run test.
        _run_result = CliRunner().invoke(
            __main__.run_full,
            [str(_input_dir)],
        )

        # 3. Verify final expectations.
        assert _run_result.exit_code == 0
        assert _output_dir.exists()
        assert any(_output_dir.glob("*"))
