import itertools
import logging
from collections import defaultdict
from operator import itemgetter

from numpy import nanmax, prod
from peewee import SqliteDatabase, fn
from scipy.interpolate import interp1d

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.assessment_mechanism_result import AssessmentMechanismResult
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


class ListOfDictToCustomMeasureExporter(OrmExporterProtocol):
    _db: SqliteDatabase

    def __init__(self, db_context: SqliteDatabase) -> None:
        if not isinstance(db_context, SqliteDatabase):
            raise ValueError(
                f"Database context ({SqliteDatabase.__name__}) required for export."
            )
        self._db = db_context

    @staticmethod
    def get_interpolated_beta_for_custom_measure(
        custom_values: list[tuple[int, float]], computation_periods: list[int]
    ) -> dict[int, float]:
        if len(custom_values) == 1:
            # If only one value is provided (0), then the rest are constant already
            return {_t: custom_values[0][1] for _t in computation_periods}

        _times, _betas = zip(*custom_values)
        _interpolate_function = interp1d(
            _times, _betas, fill_value=custom_values[-1][1]
        )
        return {
            _year: float(_interpolate_function(_year)) for _year in computation_periods
        }

    def _combine_custom_mechanism_values_to_section(
        self,
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

    def _get_grouped_dictionaries_by_measure(
        self, custom_measures: list[dict]
    ) -> dict[tuple[str, str, str], list[dict]]:
        # Unfortunately we need to check whether all groups contain a custom measure
        # for t=0, which makes the code less efficient.
        _missing_t0_measures = []
        _grouped_by_measure = defaultdict(list)
        for _measure_keys, _grouped_custom_measures in itertools.groupby(
            custom_measures,
            key=itemgetter("MEASURE_NAME", "COMBINABLE_TYPE", "SECTION_NAME"),
        ):
            _grouped_by_measure[_measure_keys] = list(_grouped_custom_measures)
            if not any(gm["TIME"] == 0 for gm in _grouped_by_measure[_measure_keys]):
                _missing_t0_measures.append(
                    f"Missing t0 beta value for Custom Measure {_measure_keys[0]} - {_measure_keys[1]} - {_measure_keys[2]}"
                )
        if any(_missing_t0_measures):
            _missing_t0_measures_str = "\n".join(_missing_t0_measures)
            raise ValueError(
                f"It was not possible to export the custom measures to the database, detailed error:\n{_missing_t0_measures_str}"
            )

        return _grouped_by_measure

    def export_dom(self, dom_model: list[dict]) -> list[CustomMeasure]:
        _measure_result_mechanism_to_add = []
        _measure_result_section_to_add = []
        _exported_measures = []
        for (
            _measure_unique_keys,
            _grouped_custom_measures,
        ) in self._get_grouped_dictionaries_by_measure(dom_model).items():
            _measure_name = _measure_unique_keys[0]
            _section_name = _measure_unique_keys[2]

            # Create the measure and as many `CustomMeasures` as required.
            _new_measure, _measure_created = Measure.get_or_create(
                name=_measure_name,
                measure_type=MeasureType.get_or_create(
                    name=MeasureTypeEnum.CUSTOM.get_old_name()
                )[0],
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

            (_retrieved_custom_measures) = self._get_custom_measures(
                _grouped_custom_measures, _new_measure
            )
            _exported_measures.extend(_retrieved_custom_measures)

            # Add the related entries in `MEASURE_RESULT`.
            (
                _mr_sections,
                _mr_mechanisms,
            ) = self._get_measure_result_section_and_mechanism(
                _retrieved_custom_measures,
                _new_measure_result,
                _new_measure_per_section,
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
    ) -> list[CustomMeasure]:
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
        return _custom_measures

    def _get_beta_cost_for_custom_measure_section(
        self,
        mechanism_per_section: MechanismPerSection,
        custom_mechanism_collection: dict,
        year: int,
    ) -> tuple[float, float]:
        # We verify whether the mechanism exists in our collection
        # directly instead of `dict.get(key, fallback)`
        # otherwise it evaluates the fallback option
        # which in our case would be an sql query (or a method),
        # either way implying extra computational cost.
        if mechanism_per_section.mechanism in custom_mechanism_collection:
            _custom_measure = custom_mechanism_collection[
                mechanism_per_section.mechanism
            ]
            return _custom_measure.beta, _custom_measure.cost

        return (
            self.get_interpolated_beta_from_assessment(mechanism_per_section, year),
            float("nan"),
        )

    def _get_mechanisms_interpolated_betas(
        self,
        custom_measures: list[CustomMeasure],
        measure_per_section: MeasurePerSection,
    ) -> dict[Mechanism, dict[int, float]]:
        _custom_mechanism_betas = dict()
        for _mechanism, _custom_measures_group in itertools.groupby(
            custom_measures, key=lambda x: x.mechanism
        ):
            _time_betas = [(_cm.year, _cm.beta) for _cm in _custom_measures_group]
            _mechanism_per_section = (
                measure_per_section.section.mechanisms_per_section.where(
                    MechanismPerSection.mechanism == _mechanism
                ).get()
            )
            _available_times = list(
                sorted(
                    _amr.time
                    for _amr in _mechanism_per_section.assessment_mechanism_results.select(
                        AssessmentMechanismResult.time
                    ).distinct()
                )
            )
            _interpolated_betas_dict = self.get_interpolated_beta_for_custom_measure(
                _time_betas, _available_times
            )
            _custom_mechanism_betas[_mechanism] = _interpolated_betas_dict
        return _custom_mechanism_betas

    def _get_measure_result_section_and_mechanism(
        self,
        custom_measures: list[CustomMeasure],
        measure_result: MeasureResult,
        measure_per_section: MeasurePerSection,
    ) -> tuple[list[dict], list[dict]]:
        """
        - Add `MeasureResultMechanism` for custom measures with and without defined mechanism.
        - Add the `MeasureResultSection` definition with all known data per `time`.
        """
        _measure_result_mechanism_to_add = []
        _measure_result_section_to_add = []
        _custom_mechanism_betas = self._get_mechanisms_interpolated_betas(
            custom_measures, measure_per_section
        )
        _cost = next((_cm for _cm in custom_measures), float("nan"))
        for (
            _mechanism_per_section
        ) in measure_per_section.section.mechanisms_per_section:
            _fallback_value = []
            if _mechanism_per_section.mechanism not in _custom_mechanism_betas:
                _fallback_value = {
                    _amr.time: _amr.beta
                    for _amr in _mechanism_per_section.assessment_mechanism_results
                }
            _measure_result_mechanism_to_add.extend(
                [
                    dict(
                        measure_result=measure_result,
                        mechanism_per_section=_mechanism_per_section,
                        time=_cm_year,
                        beta=_cm_beta,
                    )
                    for _cm_year, _cm_beta in _custom_mechanism_betas.get(
                        _mechanism_per_section.mechanism, _fallback_value
                    ).items()
                ]
            )

        # Add `MeasureResultSection` data.
        for _year, _mrm_by_year in itertools.groupby(
            sorted(_measure_result_mechanism_to_add, key=lambda x: x["time"]),
            key=lambda x: x["time"],
        ):
            _measure_result_section_to_add.append(
                dict(
                    measure_result=measure_result,
                    time=_year,
                    beta=self._combine_custom_mechanism_values_to_section(
                        [_mrm["beta"] for _mrm in _mrm_by_year]
                    ),
                    # Costs should be identical
                    # Get the maximum in case the first one was extracted from
                    # `AssessmentMechanismResult` as  `float("nan")`.
                    cost=_cost,
                )
            )
        return _measure_result_section_to_add, _measure_result_mechanism_to_add
