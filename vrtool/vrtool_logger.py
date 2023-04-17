from __future__ import annotations

import logging
from pathlib import Path


class VrToolLogger:

    @staticmethod
    def init_file_handler(log_file: Path) -> None:
        """
        Creates an empty log file, a root logger and sets the log stream to output its content to said file with the mininmum logging level.

        Args:
            log_file (Optional[Path]): Location where the log file will be saved.

        Raises:
            ValueError: When no 'log_file' argument is provided.
        """
        if not log_file:
            raise ValueError("Missing 'log_file' argument.")

        if log_file.suffix != ".log":
            log_file = log_file.with_suffix(".log")

        # Initialize log file.
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.unlink(missing_ok=True)
        log_file.touch()

        # Set file handler
        _file_handler = logging.FileHandler(filename=log_file, mode="a")
        _file_handler.setLevel(logging.INFO)
        _file_handler.setFormatter(VrToolLogger.get_vrtool_formatter())
        VrToolLogger.add_handler(_file_handler, logging.INFO)

    @staticmethod
    def init_console_handler() -> None:
        """
        Creates a console handler, the root logger and sets the minimum logging level to it.
        """
        # Set console handler
        _console_handler = logging.StreamHandler()
        _console_handler.setLevel(logging.INFO)  # Can be also set to WARNING
        _console_handler.setFormatter(VrToolLogger.get_vrtool_formatter())
        VrToolLogger.add_handler(_console_handler, logging.INFO)
        
  
    @staticmethod
    def add_handler(handler: logging.StreamHandler, logging_level: logging._Level):
        """
        Adds a new handler to the VrTool logger.

        Args:
            handler (logging.StreamHandler): Handler to be added.
        """
        # Set logger
        _logger = logging.getLogger("")
        _logger.setLevel(logging_level)
        _logger.addHandler(handler)
        logging.info(f"Initialized VrTool logger.")


    @staticmethod
    def get_vrtool_formatter() -> logging.Formatter:
        # Create a formatter and add to the file and console handlers.
        return logging.Formatter(
            fmt="%(asctime)s - [%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %I:%M:%S %p",
        )