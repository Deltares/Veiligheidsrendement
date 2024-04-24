from dataclasses import dataclass
from pathlib import Path

from pandas import read_csv


@dataclass
class MeasureUnitCosts:
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

    @classmethod
    def from_unknown_dict(cls, unknown_dict: dict):
        return cls(
            **{
                key.strip().lower().replace(" ", "_"): value
                for key, value in unknown_dict.items()
            }
        )

    @classmethod
    def from_csv_file(cls, csv_file: Path):
        """
        Returns the _default_ unit costs read from the default csv file, with columns: 'Description', 'Cost' and 'Unit'.
        Raises:
            FileNotFoundError: When the default "unit_costs.csv" file is not found.
        Returns:
            dict: Unit costs dictionary.
        """
        if not csv_file.is_file():
            raise FileNotFoundError(
                "Default unit costs file not found at {}.".format(csv_file)
            )
        _unit_cost_data = read_csv(str(csv_file), encoding="latin_1")
        _unit_cost_dict = {}
        for _, _series in _unit_cost_data.iterrows():
            _unit_cost_dict[_series["Description"]] = _series["Cost"]
        return MeasureUnitCosts.from_unknown_dict(_unit_cost_dict)
