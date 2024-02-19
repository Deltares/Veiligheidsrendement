from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.orm.io.importers.optimization.optimization_measure_result_importer import (
    OptimizationMeasureResultImporter,
)
from vrtool.orm.models.measure_result import MeasureResult as OrmMeasureResult
from vrtool.orm.models.section_data import SectionData as OrmSectionData


class OptimizationSectionAsInputImporter:
    config: VrtoolConfig

    def __init__(self, vrtool_config: VrtoolConfig) -> None:
        self.config = vrtool_config

    def import_from_section_data_results(
        self,
        section_data_results: tuple[OrmSectionData, dict[OrmMeasureResult, list[int]]],
    ) -> SectionAsInput:
        """
        Imports the filtered collection of `SectionAsInput` containing all the information related
        to the requested measure results (`MeasureResult`).

        Args:
            section_data_results (tuple[OrmSectionData, dict[OrmMeasureResult, list[int]]]): Measure results and requested years to import for a given `SectionData`.

        Returns:
            SectionAsInput: Mapped resulting object.
        """
        _imported_measures = []
        _section_data, _measure_results_dict = section_data_results
        for _measure_result, _investment_years in _measure_results_dict.items():
            _imported_measures.extend(
                OptimizationMeasureResultImporter(
                    self.config, _investment_years
                ).import_orm(_measure_result)
            )

        # TODO: Update inital costs ONLY for SoilReinforcement measures (as primary)
        # The code from the strategy controller can be used for this.

        return SectionAsInput(
            section_name=_section_data.section_name,
            traject_name=_section_data.dike_traject.traject_name,
            measures=_imported_measures,
        )
