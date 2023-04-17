import pytest
from vrtool.vrtool_logger import VrToolLogger
import logging

class TestVrToolLogger:

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
    
    def test_init_console_handler_adds_handler_to_logging(self):
        # 1. Define expectations
        # Assumes the vrtool logger is the root one 
        _vrtool_logger = logging.getLogger("")  
        assert not _vrtool_logger.handlers
        
        # 2. Run test
        VrToolLogger.init_console_handler()

        # 3. Verify expectations.
        assert len(_vrtool_logger.handlers) == 1
        _handler = _vrtool_logger.handlers[0]
        assert isinstance(_handler, logging.StreamHandler)
        assert _handler.level == logging.INFO


    @pytest.mark.parametrize("logging_level", [pytest.param(logging.INFO), pytest.param(logging.WARN), pytest.param(logging.DEBUG), pytest.param(logging.ERROR)])
    def test_add_handler_sets_the_logging_level(self, logging_level: int):
        # 1. Define test data.
        _test_handler = logging.StreamHandler()
        _test_handler.name = "MyTestHandler"
        _vrtool_logger = logging.getLogger("")  

        assert not _vrtool_logger.handlers

        # 2. Run test.
        VrToolLogger.add_handler(_test_handler, logging_level)

        # 3. Verify expectations.
        assert _test_handler in _vrtool_logger.handlers
        assert _vrtool_logger.level == logging_level
        

    