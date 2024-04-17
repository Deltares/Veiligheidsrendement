import logging
from dataclasses import dataclass, field

import pandas as pd

from vrtool.common.enums.direction_enum import DirectionEnum


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


@dataclass
class SoilReinforcementMeasureCalculator:
    section_name: str
    unit_costs: MeasureUnitCosts
    length: float
    depth: float
    dcrest: float
    dberm_in: int
    area_extra: float
    area_excavated: float
    with_stability_screen: bool = False
    housing: pd.DataFrame = field(default_factory=lambda: pd.DataFrame([]))
    direction: DirectionEnum = field(default_factory=lambda: DirectionEnum.INVALID)

    def _calculate_inward_cost(self) -> float:
        return (
            self.unit_costs.inward_added_volume * self.area_extra * self.length
            + self.unit_costs.inward_starting_costs * self.length
        )

    def _calculate_outward_cost(self) -> float:
        volume_excavated = self.area_excavated * self.length
        volume_extra = self.area_extra * self.length
        reusable_volume = self.unit_costs.outward_reuse_factor * volume_excavated
        # excavate and remove part of existing profile:
        total_cost = self.unit_costs.outward_removed_volume * (
            volume_excavated - reusable_volume
        )

        # apply reusable volume
        total_cost += self.unit_costs.outward_reused_volume * reusable_volume
        remaining_volume = volume_extra - reusable_volume

        # add additional soil:
        total_cost += self.unit_costs.outward_added_volume * remaining_volume

        # compensate:
        total_cost += (
            self.unit_costs.outward_removed_volume
            * self.unit_costs.outward_compensation_factor
            * volume_extra
        )
        return total_cost

    def _calculate_direction_costs(self) -> float:
        if (self.direction == DirectionEnum.OUTWARD) and (self.dberm_in > 0):
            # as we only use unit costs for outward reinforcement, and these are typically lower, the computation might be incorrect (too low).
            logging.warning(
                "Buitenwaartse versterking met binnenwaartse berm (dijkvak {}) kan leiden tot onnauwkeurige kostenberekeningen".format(
                    self.section_name
                )
            )
        if self.direction == DirectionEnum.INWARD:
            return self._calculate_inward_cost()
        elif self.direction == DirectionEnum.OUTWARD:
            return self._calculate_outward_cost()

        raise ValueError("Invalid direction: {}".format(self.direction))

    def _calculate_housing_costs(self) -> float:
        if self.dberm_in <= 0:
            # No costs to calculate.
            return 0.0

        if self.dberm_in > self.housing.size:
            logging.warning(
                "Binnenwaartse teenverschuiving is groter dan gegevens voor bebouwing op dijkvak {}".format(
                    self.section_name
                )
            )
            return (
                self.unit_costs.house_removal
                * self.housing.loc[self.housing.size]["cumulative"]
            )
        return (
            self.unit_costs.house_removal
            * self.housing.loc[self.dberm_in]["cumulative"]
        )

    def calculate_total_cost(self) -> float:

        _direction_costs = self._calculate_direction_costs()
        _housing_costs = self._calculate_housing_costs()

        _total_costs = _direction_costs + _housing_costs

        # add costs for stability screen
        # TODO: only passing parameters because of this.
        if self.with_stability_screen:
            _total_costs += self.unit_costs.sheetpile * self.depth * self.length

        if self.dcrest > 0.0:
            _total_costs += self.unit_costs.road_renewal * self.length

        return _total_costs
