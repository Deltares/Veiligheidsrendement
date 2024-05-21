import itertools
from collections import defaultdict

from numpy import prod
from scipy.interpolate import interp1d

from vrtool.orm.models.custom_measure import CustomMeasureDetails
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


class CustomMeasureTimeBetaCalculator:
    """
    Class to calculate time-beta values through interpolation and apply
    the defined constraints (such as last value becomes a constant for greater t values)

    !IMPORTANT!: This class is to be used within a `vrtool_db` (`SqliteDatabase`) context!
    """

    _assessment_time_beta_collection: dict[Mechanism, dict[int, float]]
    _custom_time_beta_collection: dict[Mechanism, dict[int, float]]
    _mechanism_per_section: dict[Mechanism, MechanismPerSection]
    _custom_measure_cost: float

    def __init__(
        self,
        measure_per_section: MeasurePerSection,
        custom_measures: list[CustomMeasureDetails],
    ) -> None:
        self._assessment_time_beta_collection = defaultdict(dict)
        self._custom_time_beta_collection = dict()
        self._mechanism_per_section = dict()

        # The cost is meant to be unique for all defined custom measures.
        self._custom_measure_cost = (
            float("nan") if not any(custom_measures) else custom_measures[0].cost
        )
        self._set_assessment_time_beta_collection(measure_per_section)
        self._set_custom_time_beta_collection(custom_measures)

    def _set_assessment_time_beta_collection(
        self, measure_per_section: MeasurePerSection
    ):
        for _mps in measure_per_section.section.mechanisms_per_section:
            self._mechanism_per_section[_mps.mechanism] = _mps
            self._assessment_time_beta_collection[_mps.mechanism] = {
                _amr.time: _amr.beta for _amr in _mps.assessment_mechanism_results
            }

    def _set_custom_time_beta_collection(
        self, custom_measures: list[CustomMeasureDetails]
    ):
        for _mechanism, _custom_measures_group in itertools.groupby(
            sorted(custom_measures, key=lambda x: x.mechanism.name),
            key=lambda x: x.mechanism,
        ):
            _time_betas = [(_cm.year, _cm.beta) for _cm in _custom_measures_group]
            _interpolated_betas_dict = self.get_interpolated_time_beta_collection(
                _time_betas,
                sorted(self._assessment_time_beta_collection[_mechanism].keys()),
            )
            self._custom_time_beta_collection[_mechanism] = _interpolated_betas_dict

    @staticmethod
    def get_interpolated_time_beta_collection(
        custom_values: list[tuple[int, float]], computation_periods: list[int]
    ) -> dict[int, float]:
        """
        Derives beta values for the provided `computation_periods` by interpolating
        the function of the time-beta values given as `custom_values`.
        - The last item in `custom_values` becomes a constant over time.

        Args:
            custom_values (list[tuple[int, float]]):
                List of tuples representing a year and its beta value.
            computation_periods (list[int]):
                List of years whose beta values need to be derived.

        Returns:
            dict[int, float]: Dictionary with `year` as keys and `beta` as values.
        """
        if len(custom_values) == 1:
            # If only one value is provided (0), then the rest are constant already
            return {_t: custom_values[0][1] for _t in computation_periods}

        _times, _betas = zip(*custom_values)
        _interpolate_function = interp1d(
            _times,
            _betas,
            fill_value=(custom_values[0][1], custom_values[-1][1]),
            bounds_error=False,
        )
        return {
            _year: float(_interpolate_function(_year)) for _year in computation_periods
        }

    @staticmethod
    def get_custom_mechanism_values_to_section_combination(
        mechanism_beta_values: list[float],
    ) -> float:
        """
        This method belongs in a "future" dataclass representing the
        CsvCustomMeasure File-Object-Model
        """

        def exceedance_probability_swap(value: float) -> float:
            return 1 - beta_to_pf(value)

        _product = prod(list(map(exceedance_probability_swap, mechanism_beta_values)))
        return pf_to_beta(1 - _product)

    def calculate(self, measure_result: MeasureResult) -> tuple[list[dict], list[dict]]:
        """
        Calculates the `time-beta` collections (`list[dict]`) for the
        `MeasureResultSection` and `MeasureResultMechanism` tables.

        Args:
            measure_result (MeasureResult): Foreign key to use in the collections.

        Returns:
            tuple[list[dict], list[dict]]: The first list contains dictionaries that
            can be directly used to create entries in the `MeasureResultSection` table,
              the second one does the same for the `MeasureResultMechanism` table.
        """
        _measure_result_time_beta_collection = (
            self._assessment_time_beta_collection | self._custom_time_beta_collection
        )

        _measure_result_mechanism_collection = list(
            dict(
                measure_result=measure_result,
                mechanism_per_section=self._mechanism_per_section[_mechanism],
                beta=_mr_beta,
                time=_mr_time,
            )
            for _mechanism, _time_beta_collection in _measure_result_time_beta_collection.items()
            for _mr_time, _mr_beta in _time_beta_collection.items()
        )

        # Get `MeasureResultSection` data.
        _measure_result_section_collection = list(
            dict(
                measure_result=measure_result,
                time=_year,
                beta=self.get_custom_mechanism_values_to_section_combination(
                    [_mrm["beta"] for _mrm in _mrm_by_year]
                ),
                cost=self._custom_measure_cost,
            )
            for _year, _mrm_by_year in itertools.groupby(
                sorted(_measure_result_mechanism_collection, key=lambda x: x["time"]),
                key=lambda x: x["time"],
            )
        )
        return _measure_result_section_collection, _measure_result_mechanism_collection
