from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.defaults import default_unit_costs_csv


@dataclass
class VrtoolConfig:
    """
    This is a file with all the general configuration settings for the SAFE computations
    Use them in a function by calling import config, and then config.key
    TODO: Potentially transform all fix strings values into enums (or class types).
    TODO: Refactor properties to follow python standard (snakecase)
    """

    # Directory to write the results to
    input_database_name: str = ""
    input_directory: Path = None
    output_directory: Optional[Path] = None
    externals: Optional[Path] = None  # for DStability
    language: str = "EN"

    ## RELIABILITY COMPUTATION
    traject: str = ""
    # year the computation starts
    t_0: int = 2025
    # years to compute reliability for
    T: list[int] = field(default_factory=lambda: [0, 19, 20, 25, 50, 75, 100])
    # mechanisms to exclude
    excluded_mechanisms: list[MechanismEnum] = field(
        default_factory=lambda: [
            MechanismEnum.HYDRAULIC_STRUCTURES,
        ]
    )
    # whether to consider length-effects within a dike section
    LE_in_section: bool = False
    crest_step: float = 0.5
    berm_step: list[int] = field(default_factory=lambda: [0, 5, 8, 10, 12, 15, 20, 30])

    ## OPTIMIZATION SETTINGS
    # Design horizon for TargetReliabilityBased approach
    OI_horizon: int = 50
    # Stop criterion for benefit-cost ratio
    BC_stop: float = 0.1
    # maximum number of iterations in the greedy search algorithm
    max_greedy_iterations: int = 150
    # cautiousness factor for the greedy search algorithm. Larger values result in larger steps but lower accuracy and larger probability of finding a local optimum
    f_cautious: float = 1.5
    discount_rate: float = 0.03

    ## OUTPUT SETTINGS:

    design_methods: list[str] = field(
        default_factory=lambda: ["Veiligheidsrendement", "Doorsnede-eisen"]
    )

    unit_costs: MeasureUnitCosts = field(
        default_factory=lambda: MeasureUnitCosts.from_csv_file(default_unit_costs_csv)
    )

    @property
    def input_database_path(self) -> None | Path:
        """
        Construct database path as property from input dir and database name
        """
        if not (self.input_directory and self.input_database_name):
            return None
        return self.input_directory.joinpath(self.input_database_name)

    @property
    def supported_mechanisms(self) -> list[MechanismEnum]:
        """Mechanisms that are supported"""
        return list(
            mech for mech in list(MechanismEnum) if mech != MechanismEnum.INVALID
        )

    @property
    def mechanisms(self) -> list[MechanismEnum]:
        """Filtered list of mechanisms"""

        def non_excluded_mechanisms(mechanism: MechanismEnum) -> bool:
            return mechanism not in self.excluded_mechanisms

        return list(filter(non_excluded_mechanisms, self.supported_mechanisms))

    def __post_init__(self):
        """
        After initialization and set of the values through the constructor we modify certain properties to ensure they are of the correct type.
        """

        def _convert_to_path(value: Union[None, Path, str]) -> Union[None, Path]:
            if not value:
                return None

            if isinstance(value, str):
                return Path(value)
            return value

        self.input_directory = _convert_to_path(self.input_directory)
        self.output_directory = _convert_to_path(self.output_directory)
        self.externals = _convert_to_path(self.externals)

        def _valid_mechanism(mechanism: MechanismEnum | str) -> MechanismEnum:
            if isinstance(mechanism, MechanismEnum):
                return mechanism
            return MechanismEnum.get_enum(mechanism)

        self.excluded_mechanisms = list(map(_valid_mechanism, self.excluded_mechanisms))

    def _relative_paths_to_absolute(self, parent_path: Path):
        """
        Converts all relevant relative paths to absolute paths.

        Args:
            parent_path (Path): Parent path to apply to the relative values.
        """

        def _relative_to_absolute(value: Union[None, Path]) -> Union[None, Path]:
            if not value:
                return value
            if value.is_absolute():
                return value
            return parent_path.joinpath(value).resolve()

        self.input_directory = _relative_to_absolute(self.input_directory)
        self.output_directory = _relative_to_absolute(self.output_directory)
        self.externals = _relative_to_absolute(self.externals)

    def export(self, export_path: Path):
        """
        Exports the non-default values of this configuration into a JSON file.
        Args:
            export_path (Path): Location where to export the configuration.
        """
        _default_config = VrtoolConfig().__dict__
        _custom_entries = {}
        _current_config = self.__dict__
        for _key, _default_value in _default_config.items():
            _current_value = _current_config[_key]
            if _default_value != _current_value:
                _custom_entries[_key] = _current_value

        if not export_path.parent.exists():
            export_path.parent.mkdir(parents=True)
        export_path.write_text(json.dumps(_custom_entries, sort_keys=True, indent=4))

    @classmethod
    def from_json(cls, json_path: Path) -> VrtoolConfig:
        """
        Loads all the custom properties defined in a json file into an instance of `VrtoolConfig`.
        Args:
            json_path (Path): Valid path to a json file.
        Returns:
            VrtoolConfig: Valid instance with custom configuration values.
        """
        _custom_config = json.loads(json_path.read_text())
        _vrtool_config = cls(**_custom_config)
        _vrtool_config._relative_paths_to_absolute(json_path.parent)
        return _vrtool_config
