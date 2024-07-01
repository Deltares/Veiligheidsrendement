from pathlib import Path
from typing import Protocol

from vrtool.defaults.vrtool_config import VrtoolConfig


def _get_database_reference_path(
    vrtool_config: VrtoolConfig, suffix_name: str = ""
) -> Path:
    # Get database paths.
    _reference_database_path = vrtool_config.input_database_path.with_name(
        f"vrtool_input{suffix_name}.db"
    )
    assert (
        _reference_database_path != vrtool_config.input_database_path
    ), "Reference and result database point to the same Path {}.".path(
        vrtool_config.input_database_path
    )
    return _reference_database_path


class RunStepValidator(Protocol):
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        """
        Validates the initial expectations for a specific run step.

        Args:
            valid_vrtool_config (VrtoolConfig): Configuration to be used.
        """
        pass

    def validate_results(self, valid_vrtool_config: VrtoolConfig):
        """
        Validates the final expectations for a specific run step.

        Args:
            valid_vrtool_config (VrtoolConfig): Configuration to be used.
        """
        pass
