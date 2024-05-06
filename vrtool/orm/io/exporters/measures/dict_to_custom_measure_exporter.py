import itertools
import logging
import math
from collections import defaultdict
from operator import itemgetter

from numpy import prod
from peewee import SqliteDatabase, fn
from scipy.interpolate import interp1d

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.combinable_type import CombinableType
from vrtool.orm.models.custom_measure import CustomMeasure
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.models.measure_result.measure_result_mechanism import (
    MeasureResultMechanism,
)
from vrtool.orm.models.measure_result.measure_result_section import MeasureResultSection
from vrtool.orm.models.measure_type import MeasureType
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


class DictListToCustomMeasureExporter(OrmExporterProtocol):
    _db: SqliteDatabase
    _reliability_years: list[int]

    def __init__(
        self, db_context: SqliteDatabase, reliability_years: list[int]
    ) -> None:
        self._db = db_context
        self._reliability_years = reliability_years

    @staticmethod
    def combine_custom_mechanism_values_to_section(
        mechanism_beta_values: list[float],
    ) -> float:
        """
        Combines the beta values of different mechanisms  for the same
        computation time to generate the related  `MeasureResultSection.beta`
        value.

        This method belongs in a "future" dataclass representing the
        CsvCustomMeasure File-Object-Model

        Args:
            mechanism_beta_values (list[float]):
                List of `MeasureResultMechanism.beta` values.

        Returns:
            float: The calculated `MeasureResultSection.beta` value.
        """

        def exceedance_probability_swap(value: float) -> float:
            return 1 - beta_to_pf(value)

        _product = prod(list(map(exceedance_probability_swap, mechanism_beta_values)))
        return pf_to_beta(1 - _product)

    def export_dom(self, dom_model: list[dict]) -> list[CustomMeasure]:
        _measure_result_mechanism_to_add = []
        _measure_result_section_to_add = []
        _exported_measures = []
        for _measure_unique_keys, _grouped_custom_measures in itertools.groupby(
            dom_model,
            key=itemgetter("MEASURE_NAME", "COMBINABLE_TYPE", "SECTION_NAME"),
        ):
            _measure_name = _measure_unique_keys[0]
            _section_name = _measure_unique_keys[2]

            # Create the measure and as many `CustomMeasures` as required.
            _new_measure, _measure_created = Measure.get_or_create(
                name=_measure_name,
                measure_type=MeasureType.get_or_create(name=MeasureTypeEnum.CUSTOM)[0],
                combinable_type=CombinableType.select()
                .where(
                    fn.upper(CombinableType.name)
                    == str(CombinableTypeEnum.get_enum(_measure_unique_keys[1]))
                )
                .get(),
            )
            if not _measure_created:
                logging.warning(
                    "Found existing %s measure, custom measures could be updated based on the new entries.",
                    _measure_name,
                )

            # Add entry to `MeasurePerSection`
            _new_measure_per_section, _ = MeasurePerSection.get_or_create(
                section=SectionData.get(section_name=_section_name),
                measure=_new_measure,
            )

            # Add MeasureResult
            _new_measure_result, _ = MeasureResult.get_or_create(
                measure_per_section=_new_measure_per_section
            )

            (
                _retrieved_custom_measures,
                _custom_measures_by_year,
            ) = self._get_custom_measures(_grouped_custom_measures, _new_measure)
            _exported_measures.extend(_retrieved_custom_measures)

            # Add the related entries in `MEASURE_RESULT`.
            (
                _mr_sections,
                _mr_mechanisms,
            ) = self._get_measure_result_section_and_mechanism(
                _custom_measures_by_year, _new_measure_result, _new_measure_per_section
            )
            _measure_result_section_to_add.extend(_mr_sections)
            _measure_result_mechanism_to_add.extend(_mr_mechanisms)

        # Insert bulk (more efficient) the dictionaries we just created.
        MeasureResultSection.insert_many(_measure_result_section_to_add).execute(
            self._db
        )
        MeasureResultMechanism.insert_many(_measure_result_mechanism_to_add).execute(
            self._db
        )

        return _exported_measures

    def _get_custom_measures(
        self, custom_measure_list_dict: list[dict], parent_measure: Measure
    ) -> tuple[list[CustomMeasure], dict[int, dict[Mechanism, CustomMeasure]]]:
        _custom_measures_by_year = defaultdict(dict)
        _custom_measures = []
        for _custom_measure in custom_measure_list_dict:
            _mechanism_found = (
                Mechanism.select()
                .where(
                    fn.upper(Mechanism.name)
                    == str(MechanismEnum.get_enum(_custom_measure["MECHANISM_NAME"]))
                )
                .get()
            )
            # This is not the most efficient way, but it guarantees previous custom measures
            # remain in place.
            _new_custom_measure, _is_new = CustomMeasure.get_or_create(
                measure=parent_measure,
                mechanism=_mechanism_found,
                cost=_custom_measure["COST"],
                beta=_custom_measure["BETA"],
                year=_custom_measure["TIME"],
            )
            if not _is_new:
                logging.info(
                    "An existing `CustomMeasure` was found for %s, no new entry will be created",
                    parent_measure.name,
                )
            _custom_measures.append(_new_custom_measure)
            _custom_measures_by_year[_new_custom_measure.year][
                _mechanism_found
            ] = _new_custom_measure
        return _custom_measures, _custom_measures_by_year

    def _get_measure_result_section_and_mechanism(
        self,
        custom_measures_by_year: dict[int, dict],
        measure_result: MeasureResult,
        measure_per_section: MeasurePerSection,
    ) -> tuple[list[dict], list[dict]]:
        """
        - Add `MeasureResultMechanism` for custom measures with and without defined mechanism.
        - Add the `MeasureResultSection` definition with all known data per `time`.
        """
        _measure_result_mechanism_to_add = []
        _measure_result_section_to_add = []

        # TODO: This should be the argument being given.
        _invertedcmb = defaultdict(dict)
        for _year, _mechanism_measures in custom_measures_by_year.items():
            for _mechanism, _measures in _mechanism_measures.items():
                _invertedcmb[_mechanism][_year] = _measures

        _section_betas = defaultdict(list)
        _section_cost = next(
            (
                _measure.cost
                for _mech_dict in _invertedcmb.values()
                for _measure in _mech_dict.values()
            ),
            float("nan"),
        )
        if math.isnan(_section_cost):
            # Cost is  supposed to be the same for all CustomMeasures
            # with the same MeasurePerSection (only expected change in time)
            logging.warning(
                "No cost was found for results related to measure %s and section %s",
                measure_per_section.measure.name,
                measure_per_section.section.section_name,
            )
        for (
            _mechanism_per_section
        ) in measure_per_section.section.mechanisms_per_section:
            _beta_per_year_dict = {
                _amr.time: _amr.beta
                for _amr in _mechanism_per_section.assessment_mechanism_results
            }
            if sorted(_beta_per_year_dict.keys()) != sorted(self._reliability_years):
                _assessment_years_str = ", ".join(_beta_per_year_dict.keys())
                _reliability_years_str = ", ".join(self._reliability_years)
                raise ValueError(
                    f"Export not possible as the assessment years ({_assessment_years_str}) do not much the provided reliability years ({_reliability_years_str})."
                )
            if _mechanism_per_section.mechanism in _invertedcmb.keys():
                for _year, _custom_measure in sorted(
                    _invertedcmb[_mechanism_per_section.mechanism].items()
                ):
                    # Replace the values for years that match.
                    # Because it's sorted we can simply replace the rest of the values.
                    for _assessment_year in _beta_per_year_dict.keys():
                        if _year <= _assessment_year:
                            _beta_per_year_dict[_assessment_year] = _custom_measure.beta

            # Add values to collection.
            for _amr_time, _amr_beta in _beta_per_year_dict.items():
                _section_betas[_amr_time].append(_amr_beta)
                _measure_result_mechanism_to_add.append(
                    dict(
                        measure_result=measure_result,
                        mechanism_per_section=_mechanism_per_section,
                        time=_amr_time,
                        beta=_amr_beta,
                    )
                )
        _measure_result_section_to_add.extend(
            [
                dict(
                    measure_result=measure_result,
                    time=_year,
                    beta=self.combine_custom_mechanism_values_to_section(
                        _section_mechanism_betas
                    ),
                    # Costs should be identical
                    cost=_section_cost,
                )
                for _year, _section_mechanism_betas in _section_betas.items()
            ]
        )

        return _measure_result_section_to_add, _measure_result_mechanism_to_add
