import itertools
import logging
from collections import defaultdict
from operator import itemgetter

from peewee import SqliteDatabase, fn

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.orm.io.exporters.measures.custom_measure_time_beta_calculator import (
    CustomMeasureTimeBetaCalculator,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.combinable_type import CombinableType
from vrtool.orm.models.custom_measure_detail import CustomMeasureDetail
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


class ListOfDictToCustomMeasureExporter(OrmExporterProtocol):
    """
    Exports a list of dictionaries representing a `CustomMeasureDetail` entry
    so that it also generates all related entries for the tables `Measure`,
    `MeasurePerSection`, `MeasureResult`, `MeasureResultSection` and
    `MeasureResultMechanism`.

    Constraints:
        - All `CustomMeasureDetail` dictionaries require at least an entry for t=0 for
        each provided measure/mechanism.
        - If more than one value is provided, derive values for intermediate times
        based on interpolation for values between the given values.
        - When a computation time is larger than any provided `t` value,
        then the last `t` given beta value for said `t` becomes a constant.
    """

    _db: SqliteDatabase

    def __init__(self, db_context: SqliteDatabase) -> None:
        if not isinstance(db_context, SqliteDatabase):
            raise ValueError(
                f"Database context ({SqliteDatabase.__name__}) required for export."
            )
        self._db = db_context

    def export_dom(self, dom_model: list[dict]) -> list[CustomMeasureDetail]:
        _measure_result_mechanism_to_add = []
        _measure_result_section_to_add = []
        _exported_measure_details = []
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
                    name=MeasureTypeEnum.CUSTOM.legacy_name
                )[0],
                year=0,
                combinable_type=CombinableType.select()
                .where(
                    fn.upper(CombinableType.name)
                    == CombinableTypeEnum.get_enum(_measure_unique_keys[1]).name
                )
                .get(),
            )
            if not _measure_created:
                logging.warning(
                    "Maatregel %s gevonden, custom maatregelen worden geupdated met nieuwe data.",
                    _measure_name,
                )

            # Add entry to `MeasurePerSection`
            (
                _retrieved_measure_per_section,
                _measure_per_section_created,
            ) = MeasurePerSection.get_or_create(
                section=SectionData.get(section_name=_section_name),
                measure=_new_measure,
            )

            if not _measure_per_section_created:
                logging.warning(
                    "Maatregel %s bestaat al in de database voor sectie %s, maatregel wordt niet toegevoegd. Hernoem de maatregel om te kunnen toevoegen.",
                    _measure_name,
                    _section_name,
                )
                continue

            # Add entries to `CustomMeasureDetail`
            _retrieved_custom_measure_details = self._get_custom_measure_details(
                _grouped_custom_measures,
                _new_measure,
                _retrieved_measure_per_section.section,
            )
            _exported_measure_details.extend(_retrieved_custom_measure_details)

            # Add `MeasureResult`
            _new_measure_result, _ = MeasureResult.get_or_create(
                measure_per_section=_retrieved_measure_per_section
            )

            # Calculate the related entries in `MeasureResult`.
            (_mr_sections, _mr_mechanisms,) = CustomMeasureTimeBetaCalculator(
                _retrieved_measure_per_section, _retrieved_custom_measure_details
            ).calculate(_new_measure_result)
            _measure_result_section_to_add.extend(_mr_sections)
            _measure_result_mechanism_to_add.extend(_mr_mechanisms)

        # Insert bulk (more efficient) the dictionaries we just created.
        MeasureResultSection.insert_many(_measure_result_section_to_add).execute(
            self._db
        )
        MeasureResultMechanism.insert_many(_measure_result_mechanism_to_add).execute(
            self._db
        )

        return _exported_measure_details

    def _get_dict_sorted_by(
        self, item_collection: list[dict], *keys_to_group_by: tuple
    ) -> itertools.groupby:
        _key_grouping = itemgetter(*keys_to_group_by)
        return itertools.groupby(
            sorted(item_collection, key=_key_grouping),
            key=_key_grouping,
        )

    def _get_grouped_dictionaries_by_measure(
        self, custom_measures: list[dict]
    ) -> dict[tuple[str, str, str], list[dict]]:
        # Unfortunately we need to check whether all groups contain a custom measure
        # for t=0, which makes the code less efficient.
        _missing_t0_measures = []
        _grouped_by_measure = defaultdict(list)
        for _measure_keys, _grouped_custom_measures in self._get_dict_sorted_by(
            custom_measures, "MEASURE_NAME", "COMBINABLE_TYPE", "SECTION_NAME"
        ):
            _grouped_by_measure[_measure_keys] = list(_grouped_custom_measures)
            for _mechanism_key, _grouped_by_mechanism in self._get_dict_sorted_by(
                _grouped_by_measure[_measure_keys], "MECHANISM_NAME"
            ):
                if not any(gm["TIME"] == 0 for gm in _grouped_by_mechanism):
                    _measure_keys_str = " - ".join(
                        list(map(str, _measure_keys)) + [_mechanism_key]
                    )
                    _missing_t0_measures.append(
                        f"Missing t0 beta value for Custom Measure {_measure_keys_str}"
                    )
        if any(_missing_t0_measures):
            _missing_t0_measures_str = "\n".join(_missing_t0_measures)
            raise ValueError(
                f"It was not possible to export the custom measures to the database, detailed error:\n{_missing_t0_measures_str}"
            )

        return _grouped_by_measure

    def _get_custom_measure_details(
        self,
        custom_measure_list_dict: list[dict],
        parent_measure: Measure,
        section_for_measure: SectionData,
    ) -> list[CustomMeasureDetail]:
        _custom_measures = []
        for _custom_measure in custom_measure_list_dict:
            _mechanism_name = MechanismEnum.get_enum(
                _custom_measure["MECHANISM_NAME"]
            ).legacy_name
            _mechanism_per_section_found = MechanismPerSection.get_or_none(
                (MechanismPerSection.section == section_for_measure)
                & (
                    MechanismPerSection.mechanism
                    == Mechanism.get_or_none(
                        fn.upper(Mechanism.name) == _mechanism_name.upper()
                    )
                )
            )
            if (
                _mechanism_per_section_found is None
                or _mechanism_per_section_found.mechanism is None
            ):
                # TODO: Untested
                logging.error(
                    "Mechanism %s bestaat niet voor sectie %s",
                    _custom_measure["MECHANISM_NAME"],
                    section_for_measure.section_name,
                )
                continue

            # This is not the most efficient way, but it guarantees previous custom measures
            # remain in place.
            _new_custom_measure, _is_new = CustomMeasureDetail.get_or_create(
                measure=parent_measure,
                mechanism_per_section=_mechanism_per_section_found,
                cost=_custom_measure["COST"],
                beta=_custom_measure["BETA"],
                year=_custom_measure["TIME"],
            )
            if not _is_new:
                logging.info(
                    "An existing `CustomMeasureDetail` was found for %s, no new entry will be created",
                    parent_measure.name,
                )
            _custom_measures.append(_new_custom_measure)
        return _custom_measures
