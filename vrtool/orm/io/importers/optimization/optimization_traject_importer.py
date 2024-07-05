from collections import defaultdict

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.orm.io.importers.optimization.optimization_section_as_input_importer import (
    OptimizationSectionAsInputImporter,
)
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.models.section_data import SectionData


class OptimizationTrajectImporter(OrmImporterProtocol):
    """
    Importer responsible to convert the given traject (selected) measure results
    in a collection of `SectionAsInput`, including sections where no measure has
    been selected (VRTOOL-561).
    """

    def __init__(
        self,
        vrtool_config: VrtoolConfig,
        measure_results_to_import: list[tuple[int, int]],
    ) -> None:
        self._vrtool_config = vrtool_config
        self._measure_results_to_import = measure_results_to_import

    def _get_measure_results_to_import(
        self, dike_traject_info: DikeTrajectInfo
    ) -> (dict[SectionData, dict[MeasureResult, list[int]]]):
        """
        Returns a dictionary of `orm.SectionData` containing dictionaries of their
        to-be-imported `orm.MeasureResult` with their respective `investment_year`.
        """
        _section_measure_result_dict = {
            _sd: defaultdict(list)
            for _sd in dike_traject_info.dike_sections
            if _sd.in_analysis
        }
        for _result_tuple in self._measure_results_to_import:
            _measure_result = MeasureResult.get_by_id(_result_tuple[0])
            _measure_section = _measure_result.measure_per_section.section
            # Should we allow importing of measures whose section is not included in the analysis?
            _section_measure_result_dict[_measure_section][_measure_result].append(
                _result_tuple[1]
            )

        return _section_measure_result_dict

    def import_orm(self, orm_model: DikeTrajectInfo) -> list[SectionAsInput]:
        _importer = OptimizationSectionAsInputImporter(self._vrtool_config)
        _section_as_input_collection = list(
            map(
                _importer.import_from_section_data_results,
                self._get_measure_results_to_import(orm_model).items(),
            )
        )
        return _section_as_input_collection
