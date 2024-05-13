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
    def get_interpolated_beta_from_assessment(
        mechanism_per_section: MechanismPerSection, year: int
    ) -> float:
        """
        Interpolates the values found in
        `MechanismPerSection.assessment_mechanism_results`
        with the provided year.

        Required by VRTOOL-506: Custom measure with T not present in assessment.

        Args:
            mechanism_per_section (MechanismPerSection): Table with assessments.
            year (int): Year to interpolate.

        Returns:
            float: The interpolated value.
        """
        _times, _betas = zip(
            *(
                (_amr.time, _amr.beta)
                for _amr in mechanism_per_section.assessment_mechanism_results
            )
        )

        return float(interp1d(_times, _betas, fill_value=("extrapolate"))(year))

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
        for (
            _year,
            _added_cm_mechanism_year_beta,
        ) in custom_measures_by_year.items():
            _section_mechanism_betas = []
            _section_mechanism_costs = []
            for (
                _mechanism_per_section
            ) in measure_per_section.section.mechanisms_per_section:

                (
                    _mechanism_beta,
                    _mechanism_cost,
                ) = self._get_beta_cost_for_custom_measure_section(
                    _mechanism_per_section, _added_cm_mechanism_year_beta, _year
                )
                _section_mechanism_betas.append(_mechanism_beta)
                _section_mechanism_costs.append(_mechanism_cost)
                _measure_result_mechanism_to_add.append(
                    dict(
                        measure_result=measure_result,
                        mechanism_per_section=_mechanism_per_section,
                        time=_year,
                        beta=_mechanism_beta,
                    )
                )

            # Add `MeasureResultSection` data.
            _measure_result_section_to_add.append(
                dict(
                    measure_result=measure_result,
                    time=_year,
                    beta=self._combine_custom_mechanism_values_to_section(
                        _section_mechanism_betas
                    ),
                    # Costs should be identical
                    # Get the maximum in case the first one was extracted from
                    # `AssessmentMechanismResult` as  `float("nan")`.
                    cost=nanmax(_section_mechanism_costs),
                )
            )
        return _measure_result_section_to_add, _measure_result_mechanism_to_add
