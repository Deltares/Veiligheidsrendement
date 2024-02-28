import logging
import shutil

import pytest

from tests import test_results
from vrtool.vrtool_logger import VrToolLogger


class TestVrToolLogger:
    @pytest.fixture(autouse=True)
    def cleanup_logging(self):
        """
        Fixture to make sure the logging is "restarted" before and after each test.
        """
        # Before the test is run.
        logging.getLogger("").handlers.clear()

        # Run test.
        yield

        # After the test is run.
        logging.getLogger("").handlers.clear()

    @pytest.mark.parametrize(
        "log_file",
        [pytest.param("", id="Empty string"), pytest.param(None, id="None value")],
    )
    def test_init_file_handler_given_no_argument_raises_error(self, log_file: str):
        # 1. Define expectations.
        _error_mssg = "Missing 'log_file' argument."

        # 2. Run test.
        with pytest.raises(ValueError) as exception_error:
            VrToolLogger.init_file_handler(log_file, logging.INFO)

        # 3. Verify expectations.
        assert str(exception_error.value) == _error_mssg

    def test_init_file_handler_adds_handler_to_logging(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define expectations
        _logging_level = logging.INFO
        _handler_name = "VrTool log file handler"
        _log_file_path = test_results / request.node.name / "vrtool.log"
        if _log_file_path.exists():
            shutil.rmtree(_log_file_path.parent)

        # Assumes the vrtool logger is the root one
        _vrtool_logger = logging.getLogger("")
        assert not any(
            _handler.name == _handler_name for _handler in _vrtool_logger.handlers
        )

        # 2. Run test
        VrToolLogger.init_file_handler(_log_file_path, _logging_level)

        # 3. Verify expectations.
        _console_handler = next(
            (
                _handler
                for _handler in _vrtool_logger.handlers
                if _handler.name == _handler_name
            ),
            None,
        )
        assert isinstance(_console_handler, logging.FileHandler)
        assert _console_handler.level == _logging_level
        assert _vrtool_logger.level == _logging_level

    def test_init_file_handler_creates_file_and_directories(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _log_file_path = test_results / request.node.name / "vrtool.log"
        if _log_file_path.exists():
            shutil.rmtree(_log_file_path.parent)

        # 2. Run test.
        VrToolLogger.init_file_handler(_log_file_path, logging.INFO)

        # 3. Verify final expectations.
        assert _log_file_path.parent.exists()
        assert _log_file_path.exists()

    def test_init_file_handler_adds_suffix_if_missing(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _log_file_path = test_results / request.node.name / "test_log"
        if _log_file_path.parent.exists():
            shutil.rmtree(_log_file_path.parent)

        # 2. Run test.
        VrToolLogger.init_file_handler(_log_file_path, logging.INFO)

        # 3. Verify final expectations.
        assert _log_file_path.parent.exists()
        assert not _log_file_path.exists()
        assert _log_file_path.with_suffix(".log").exists()

    def test_init_console_handler_adds_handler_to_logging(self):
        # 1. Define expectations
        _handler_name = "VrTool log console handler"
        _logging_level = logging.INFO
        # Assumes the vrtool logger is the root one
        _vrtool_logger = logging.getLogger("")
        assert not any(
            _handler.name == _handler_name for _handler in _vrtool_logger.handlers
        )

        # 2. Run test
        VrToolLogger.init_console_handler(_logging_level)

        # 3. Verify expectations.
        _console_handler = next(
            (
                _handler
                for _handler in _vrtool_logger.handlers
                if _handler.name == _handler_name
            ),
            None,
        )
        assert isinstance(_console_handler, logging.StreamHandler)
        assert _console_handler.level == _logging_level
        assert _vrtool_logger.level == _logging_level

    @pytest.mark.parametrize(
        "logging_level",
        [
            pytest.param(logging.INFO, id="INFO"),
            pytest.param(logging.WARN, id="WARN"),
            pytest.param(logging.DEBUG, id="DEBUG"),
            pytest.param(logging.ERROR, id="ERROR"),
        ],
    )
    def test_add_handler_sets_the_logging_level(self, logging_level: int):
        # 1. Define test data.
        _test_handler = logging.StreamHandler()
        _test_handler.name = "MyTestHandler"
        _vrtool_logger = logging.getLogger("")

        assert _test_handler not in _vrtool_logger.handlers

        # 2. Run test.
        VrToolLogger.add_handler(_test_handler, logging_level)

        # 3. Verify expectations.
        assert _test_handler in _vrtool_logger.handlers
        assert _vrtool_logger.level == logging_level

    def test_add_handler_logs_info_message(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _log_file_path = test_results / request.node.name / "vrtool.log"
        if _log_file_path.parent.exists():
            shutil.rmtree(_log_file_path.parent)
        _log_file_path.parent.mkdir(parents=True, exist_ok=True)
        _log_file_path.touch()

        _test_handler = logging.FileHandler(filename=_log_file_path, mode="a")
        _test_handler.name = "MyFileTestHandler"
        _expected_log_mssg = "Test logging message"

        _vrtool_logger = logging.getLogger("")

        assert _test_handler not in _vrtool_logger.handlers

        # 2. Run test.
        VrToolLogger.add_handler(_test_handler, logging.DEBUG)

        # 3. Verify expectations.
        _log_lines = _log_file_path.read_text().splitlines()
        assert len(_log_lines) == 1
        assert _expected_log_mssg in _log_lines[0]
