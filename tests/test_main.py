import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests import test_data, test_results
from vrtool import __main__
from vrtool.defaults.vrtool_config import VrtoolConfig


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
            "input_database_path": str(_input_dir / "vrtool_input.db"),
            "traject": "38-1",
            "output_directory": str(_output_dir),
            "mechanisms": ["Overflow", "StabilityInner", "Piping"],
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

    def test_given_directory_without_json_raises_error(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _input_dir = test_results / request.node.name
        if not _input_dir.exists():
            _input_dir.mkdir(parents=True)

        assert _input_dir.exists()

        # 2. Run test.
        with pytest.raises(FileNotFoundError) as exception_error:
            __main__._get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert str(
            exception_error.value
        ) == "No json config file found in the model directory. {}".format(_input_dir)

    def test_given_directory_with_too_many_jsons_raises_error(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _input_dir = test_results / request.node.name
        if _input_dir.exists():
            shutil.rmtree(_input_dir)

        _input_dir.mkdir(parents=True)
        Path.joinpath(_input_dir, "first.json").touch()
        Path.joinpath(_input_dir, "second.json").touch()

        # 2. Run test.
        with pytest.raises(ValueError) as exception_error:
            __main__._get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert str(
            exception_error.value
        ) == "More than one json file found in the directory {}. Only one json at the root directory supported.".format(
            _input_dir
        )

    def test_given_directory_with_valid_config_returns_vrtool_config(self):
        # 1. Define test data.
        _input_dir = test_data / "vrtool_config"
        assert _input_dir.exists()

        # 2. Run test.
        _vrtool_config = __main__._get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert isinstance(_vrtool_config, VrtoolConfig)
        assert _vrtool_config.traject == "MyCustomTraject"
        assert _vrtool_config.input_directory == _input_dir
        assert _vrtool_config.output_directory == _input_dir / "results"
