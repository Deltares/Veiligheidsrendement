import shutil
from pathlib import Path

from click.testing import CliRunner

from src import __main__
from src.defaults.vrtool_config import VrtoolConfig
from tests import test_data, test_results
import pytest

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

    def test_given_valid_path_without_config_when_run_full_then_fails(self, request: pytest.FixtureRequest):
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

    def test_given_valid_input_when_run_full_then_succeeds(self):
        # TODO: Ideally we want a really small test.
        # 1. Define test data.
        _input_dir = test_data / "integrated_SAFE_16-3_small"
        assert _input_dir.exists()

        # Ensure we have a clean results dir.
        _results_dir = test_results / "acceptance"
        if _results_dir.exists():
            shutil.rmtree(_results_dir)
        _results_dir.mkdir(parents=True)

        # 2. Run test.
        _run_result = CliRunner().invoke(
            __main__.run_full,
            [str(_input_dir)],
        )

        # 3. Verify final expectations.
        assert _run_result.exit_code == 0

    def test_given_directory_without_json_raises_error(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _input_dir = test_results / request.node.name
        if not _input_dir.exists():
            _input_dir.mkdir(parents=True)

        assert _input_dir.exists()

        # 2. Run test.
        with pytest.raises(FileNotFoundError) as exc_err:
            __main__._get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert str(exc_err.value) == "No json config file found in the model directory. {}".format(_input_dir)
    
    def test_given_directory_with_too_many_jsons_raises_error(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _input_dir = test_results / request.node.name
        if _input_dir.exists():
            shutil.rmtree(_input_dir)
        
        _input_dir.mkdir(parents=True)
        Path.joinpath(_input_dir, "first.json").touch()
        Path.joinpath(_input_dir, "second.json").touch()

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            __main__._get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert str(exc_err.value) == "More than one json file found in the directory {}. Only one json at the root directory supported.".format(_input_dir)
    
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