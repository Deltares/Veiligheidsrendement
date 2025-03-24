import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
import re

from pandas import read_csv


@dataclass
class MeasureUnitCosts:
    """
    Dataclass to represent the contents of a / the `unit_costs.csv` file.
    """

    # Inward costs
    inward_added_volume: float
    inward_starting_costs: float

    # Outward costs
    outward_reuse_factor: float
    outward_removed_volume: float
    outward_reused_volume: float
    outward_added_volume: float
    outward_compensation_factor: float

    # Other
    house_removal: float
    road_renewal: float

    # Optionals
    sheetpile: float = float("nan")
    diaphragm_wall: float = float("nan")
    vertical_geotextile: float = float("nan")
    course_sand_barrier: float = float("nan")
    anchored_sheetpile: float = float("nan")
    heavescreen: float = float("nan")
    remove_block_revetment: float = float("nan")
    remove_asphalt_revetment: float = float("nan")
    installation_of_blocks: dict[float, float] = field(default_factory=dict)

    @classmethod
    def from_unformatted_dict(cls, unformatted_dict: dict):
        """
        Generates a `MeasureUnitCosts` by formatting the keys of the provided
        dictionary into an expected field name.

        Args:
            unformatted_dict (dict): Dictionary containing the unformatted field names.

        Returns:
            MeasureUnitCosts: Resulting mapped instance.
        """

        def normalize_key_name(key_name: str) -> str:
            return key_name.strip().lower().replace(" ", "_")

        _existing_fields = [_field.name for _field in fields(cls)]
        _normalized_dict = {}

        _block_dict = {}
        for _block_key in filter(lambda x: 'installation of blocks' in x.lower(), unformatted_dict.keys()):
            _thickness = [float(number) for number in re.findall(r'\d+\.?\d*', _block_key)]
            if len(_thickness) != 1:
                raise ValueError(f"Blokdikte niet gevonden voor key: {_block_key} in unit_costs.csv. Meerdere getalswaarden gevonden, check de consistentie.")
            _block_dict[float(_thickness[0])] = unformatted_dict[_block_key]

        unformatted_dict["Installation of blocks"] = _block_dict
        #logging TODO
        
        for key, value in unformatted_dict.items():
            _normalized_key = normalize_key_name(key)
            if _normalized_key not in _existing_fields:
                logging.warning(
                    "Measure {%s} is not internally defined and won't be imported.", key
                )
                continue
            _normalized_dict[_normalized_key] = value

        return cls(**_normalized_dict)

    @classmethod
    def from_csv_file(cls, csv_file: Path):
        """
        Generates a `MeasureUnitCosts` instance from a `csv_file` with the columns: 'Description', 'Cost' and 'Unit'.

        Raises:
            FileNotFoundError: When the provided `csv_file` is not found.

        Args:
            csv_file (Path): `*.csv` file

        Returns:
            MeasureUnitCosts: Resulting mapped instance.
        """
        if not csv_file.is_file():
            raise FileNotFoundError(
                "Default unit costs file not found at {}.".format(csv_file)
            )
        _unit_cost_data = read_csv(str(csv_file), encoding="latin_1")
        _unit_cost_dict = {}
        for _, _series in _unit_cost_data.iterrows():
            _unit_cost_dict[_series["Description"]] = _series["Cost"]

        return cls.from_unformatted_dict(_unit_cost_dict)
