import shutil
import pytest
from vrtool.vrtool_logger import VrToolLogger
import logging
from tests import test_results

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
        [
            pytest.param("", id="Empty string"),
            pytest.param(None, id="None value")
        ]
    )
    def test_init_file_handler_given_no_argument_raises_error(self, log_file: str):
        # 1. Define expectations.
        _error_mssg = "Missing 'log_file' argument."

        # 2. Run test.
        with pytest.raises(ValueError) as exception_error:
            VrToolLogger.init_file_handler(log_file)

        # 3. Verify expectations.
        assert str(exception_error.value) == _error_mssg

    def test_init_file_handler_adds_handler_to_logging(self, request: pytest.FixtureRequest):
        # 1. Define expectations
        _handler_name = "VrTool log file handler"
        _log_file = test_results / request.node.name / "vrtool.log"
        if _log_file.exists():
            shutil.rmtree(_log_file.parent)

        # Assumes the vrtool logger is the root one 
        _vrtool_logger = logging.getLogger("")  
        assert not any(_handler.name == _handler_name for _handler in _vrtool_logger.handlers)
        
        # 2. Run test
        VrToolLogger.init_file_handler(_log_file)

        # 3. Verify expectations.
        _console_handler = next((_handler for _handler in _vrtool_logger.handlers if _handler.name == _handler_name), None)
        assert isinstance(_console_handler, logging.FileHandler)
        assert _console_handler.level == logging.INFO

    def test_init_file_handler_creates_file_and_directories(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _log_file = test_results / request.node.name / "vrtool.log"
        if _log_file.exists():
            shutil.rmtree(_log_file.parent)

        # 2. Run test.
        VrToolLogger.init_file_handler(_log_file)

        # 3. Verify final expectations.
        assert _log_file.parent.exists()
        assert _log_file.exists()

    def test_init_file_handler_adds_suffix_if_missing(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _log_file = test_results / request.node.name / "test_log"
        if _log_file.parent.exists():
            shutil.rmtree(_log_file.parent)

        # 2. Run test.
        VrToolLogger.init_file_handler(_log_file)

        # 3. Verify final expectations.
        assert _log_file.parent.exists()
        assert not _log_file.exists()
        assert _log_file.with_suffix(".log").exists()
    
    def test_init_console_handler_adds_handler_to_logging(self):
        # 1. Define expectations
        _handler_name = "VrTool log console handler"
        # Assumes the vrtool logger is the root one 
        _vrtool_logger = logging.getLogger("")  
        assert not any(_handler.name == _handler_name for _handler in _vrtool_logger.handlers)
        
        # 2. Run test
        VrToolLogger.init_console_handler()

        # 3. Verify expectations.
        _console_handler = next((_handler for _handler in _vrtool_logger.handlers if _handler.name == _handler_name), None)
        assert isinstance(_console_handler, logging.StreamHandler)
        assert _console_handler.level == logging.INFO


    @pytest.mark.parametrize("logging_level", [pytest.param(logging.INFO, id="INFO"), pytest.param(logging.WARN, id="WARN"), pytest.param(logging.DEBUG, id="DEBUG"), pytest.param(logging.ERROR, id="ERROR")])
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
        

    def test_get_vrtool_formatter(self):
        _formatter = VrToolLogger.get_vrtool_formatter()
        assert isinstance(_formatter, logging.Formatter)