from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.defaults import default_unit_costs_csv


def _load_default_unit_costs() -> dict:
    """
    Returns the _default_ unit costs read from the default csv file, with columns: 'Description', 'Cost' and 'Unit'.

    Raises:
        FileNotFoundError: When the default "unit_costs.csv" file is not found.

    Returns:
        dict: Unit costs dictionary.
    """
    if not default_unit_costs_csv.is_file():
        raise FileNotFoundError(
            "Default unit costs file not found at {}.".format(default_unit_costs_csv)
        )
    _unit_cost_data = pd.read_csv(str(default_unit_costs_csv), encoding="latin_1")
    unit_cost = {}
    for _, _series in _unit_cost_data.iterrows():
        unit_cost[_series["Description"]] = _series["Cost"]
    return unit_cost


@dataclass
class VrtoolConfig:
    """
    This is a file with all the general configuration settings for the SAFE computations
    Use them in a function by calling import config, and then config.key
    TODO: Potentially transform all fix strings values into enums (or class types).
    TODO: Refactor properties to follow python standard (snakecase)
    """

    # Directory to write the results to
    directory: Optional[Path] = None
    language: str = "EN"
    timing: bool = False
    input_directory: Optional[Path] = None

    ## RELIABILITY COMPUTATION
    # year the computation starts
    t_0: int = 2025
    # years to compute reliability for
    T: list[int] = field(default_factory=lambda: [0, 19, 20, 25, 50, 75, 100])
    # mechanisms to consider
    mechanisms: list[str] = field(
        default_factory=lambda: [
            "Overflow",
            "StabilityInner",
            "Piping",
        ]
    )
    # whether to consider length-effects within a dike section
    LE_in_section: bool = False
    crest_step: float = 0.5
    berm_step: list[int] = field(default_factory=lambda: [0, 5, 8, 10, 12, 15, 20, 30])

    ## OPTIMIZATION SETTINGS
    # investment year for TargetReliabilityBased approach
    OI_year: int = 0
    # Design horizon for TargetReliabilityBased approach
    OI_horizon: int = 50
    # Stop criterion for benefit-cost ratio
    BC_stop: float = 0.1
    # maximum number of iterations in the greedy search algorithm
    max_greedy_iterations: int = 150
    # cautiousness factor for the greedy search algorithm. Larger values result in larger steps but lower accuracy and larger probability of finding a local optimum
    f_cautious: float = 1.5

    ## OUTPUT SETTINGS:
    # General settings:
    shelves: bool = False  # setting to shelve intermediate results
    reuse_output: bool = False  # reuse intermediate result if available
    # whether to use 'beta' or 'prob' for plotting reliability
    beta_or_prob: str = "beta"

    # Settings for step 1:
    # Setting to turn on plotting the reliability in time for each section.
    plot_reliability_in_time: bool = False
    # Setting to turn on plotting beta of measures at each section.
    plot_measure_reliability: bool = False

    # Setting to flip the direction of the longitudinal plots. Used for SAFE as sections are numbered east-west
    flip_traject: bool = True
    # years (relative to t_0) to plot the reliability
    assessment_plot_years: list[int] = field(
        default_factory=lambda: [
            0,
            20,
            50,
        ]
    )

    # Settings for step 2:
    # Setting to plot the change in geometry for each soil reinforcement combination. Only use for debugging: very time consuming.
    geometry_plot: bool = False

    # Settings for step 3:
    # dictionary with settings for beta-cost curve:
    beta_cost_settings: dict = field(
        default_factory=lambda: {
            # whether to include symbols in the beta-cost curve
            "symbols": True,
            # base size of markers.
            "markersize": 10,
        }
    )
    unit_costs: dict = field(default_factory=lambda: _load_default_unit_costs())
