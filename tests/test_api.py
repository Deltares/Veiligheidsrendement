from vrtool.defaults.vrtool_config import VrtoolConfig
import pytest
from tests import test_data, test_results
from vrtool.api import get_valid_vrtool_config
import shutil
from pathlib import Path


class TestApi:
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
            get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert str(
            exception_error.value
        ) == "No json config file found in the model directory {}.".format(_input_dir)

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
            get_valid_vrtool_config(_input_dir)

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
        _vrtool_config = get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert isinstance(_vrtool_config, VrtoolConfig)
        assert _vrtool_config.traject == "MyCustomTraject"
        assert _vrtool_config.input_directory == _input_dir
        assert _vrtool_config.output_directory == _input_dir / "results"
