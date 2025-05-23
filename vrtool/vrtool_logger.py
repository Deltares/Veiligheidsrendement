from __future__ import annotations

import logging
from pathlib import Path


class VrToolLogger:
    @staticmethod
    def init_file_handler(log_file: Path, logging_level: int) -> None:
        """
        Creates an empty log file, a root logger and sets the log stream to output its content to said file with the mininmum logging level.

        Args:
            log_file (Path): Location where the log file will be saved.
            logging_level (int): Logging level.

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
        _file_handler.set_name("VrTool log file handler")
        VrToolLogger.add_handler(_file_handler, logging_level)

    @staticmethod
    def init_console_handler(logging_level: int) -> None:
        """
        Creates a console handler, the root logger and sets the minimum logging level to it.

        Args:
            logging_level (int): Logging level.
        """
        # Set console handler
        _console_handler = logging.StreamHandler()
        _console_handler.set_name("VrTool log console handler")
        VrToolLogger.add_handler(_console_handler, logging_level)

    @staticmethod
    def add_handler(handler: logging.StreamHandler, logging_level: int):
        """
        Adds a new handler to the VrTool logger with VrTool's custom formatter and the given logging level.

        Args:
            handler (logging.StreamHandler): Handler to be added.
            logging_level (int): Logging level.
        """
        # Set formatter.
        _formatter = logging.Formatter(
            fmt="%(asctime)s - [%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %I:%M:%S %p",
        )
        handler.setFormatter(_formatter)
        handler.setLevel(logging_level)

        # Set (root) logger.
        _logger = logging.getLogger("")
        _logger.setLevel(logging_level)
        _logger.addHandler(handler)
        logging.debug("Initialized VrTool logger with handler %s.", handler.name)
