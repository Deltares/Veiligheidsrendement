import itertools
import logging

from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure import (
    RevetmentMeasure,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result_collection import (
    RevetmentMeasureResultCollection,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_section_reliability import (
    RevetmentMeasureSectionReliability,
)
from vrtool.decision_making.measures.standard_measures.soil_reinforcement_measure import (
    SoilReinforcementMeasure,
)
from vrtool.decision_making.measures.standard_measures.stability_screen_measure import (
    StabilityScreenMeasure,
)
from vrtool.decision_making.measures.standard_measures.vertical_geotextile_measure import (
    VerticalGeotextileMeasure,
)
from vrtool.decision_making.measures.standard_measures.wall_measures.diaphragm_wall_measure import (
    DiaphragmWallMeasure,
)
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.measures.measure_importer import MeasureImporter
from vrtool.orm.io.importers.measures.measure_result_importer import (
    MeasureResultImporter,
)
from vrtool.orm.io.importers.measures.solutions_importer import SolutionsImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.measure_result.measure_result import MeasureResult


class SolutionsForMeasureResultsImporter(OrmImporterProtocol):
    def __init__(self, vrtool_config: VrtoolConfig, dike_section: DikeSection) -> None:
        if not vrtool_config:
            raise ValueError("{} not provided.".format(VrtoolConfig.__name__))
        if not dike_section:
            raise ValueError("{} not provided.".format(DikeSection.__name__))

        self._config = vrtool_config
        self._dike_section = dike_section

    def _set_measure_results(
        self, measure: MeasureProtocol, measure_results: list[MeasureResult]
    ) -> None:
        _mr_importer = MeasureResultImporter()
        _imported_results = list(map(_mr_importer.import_orm, measure_results))

        if isinstance(measure, SoilReinforcementMeasure):
            for _res_dict in _imported_results:
                _res_dict["dcrest"] = _res_dict["imported_parameters"]["dcrest"]
                _res_dict["dberm"] = _res_dict["imported_parameters"]["dberm"]
                _res_dict["StabilityScreen"] = measure.parameters["StabilityScreen"]
                if measure.parameters["StabilityScreen"] == "yes":
                    if "l_stab_screen" in measure.parameters:
                        _res_dict["l_stab_screen"] = measure.parameters["l_stab_screen"]
                    else:
                        _res_dict["l_stab_screen"] = 3.0
            measure.measures = _imported_results
            return

        if isinstance(measure, RevetmentMeasure):
            _measure_collection = RevetmentMeasureResultCollection()
            for _imported_measure in _imported_results:
                _converted_measure = RevetmentMeasureSectionReliability()
                _converted_measure.beta_target = _imported_measure[
                    "imported_parameters"
                ]["beta_target"]
                _converted_measure.transition_level = _imported_measure[
                    "imported_parameters"
                ]["transition_level"]
                _converted_measure.section_reliability = _imported_measure[
                    "Reliability"
                ]
                _converted_measure.cost = _imported_measure["Cost"]
                _converted_measure.measure_id = _imported_measure["measure_id"]
                _converted_measure.reinforcement_type = measure.parameters["Type"]
                _converted_measure.combinable_type = measure.parameters["Class"]
                _converted_measure.measure_year = measure.parameters["year"]
                _measure_collection.result_collection.append(_converted_measure)
            measure.measures = _measure_collection
            return

        if len(_imported_results) > 1:
            logging.error("Only first record will be set as for the measures.")

        measure.measures = _imported_results[0]
        if isinstance(measure, DiaphragmWallMeasure):
            # NOTE: This check also includes `AnchoredSheetpileMeasure`
            # as it implements the `DiaphragmWallMeasure`
            measure.measures["DiaphragmWall"] = "yes"
        elif isinstance(measure, VerticalGeotextileMeasure):
            measure.measures["VZG"] = "yes"
        elif isinstance(measure, StabilityScreenMeasure):
            measure.measures["Stability Screen"] = "yes"

    def import_orm(self, measure_results: list[MeasureResult]) -> Solutions:
        """
        Imports all the measures related to the given measure_results, assuming
        they belong to the same dike section.

        Args:
            measure_results (list[MeasureResult]): list of measure results which
            will be imported into a `Solutions` instance.

        Returns:
            Solutions: instance containing all measures-per-dikesection results.
        """

        _solutions = Solutions(self._dike_section, self._config)
        _solutions.measures = []

        _measure_importer = MeasureImporter(self._config)
        for _measure, _grouped_measure_results in itertools.groupby(
            measure_results, lambda x: x.measure_per_section.measure
        ):
            _imported_measure = _measure_importer.import_orm(_measure)
            self._set_measure_results(_imported_measure, _grouped_measure_results)
            _solutions.measures.append(_imported_measure)

        SolutionsImporter.set_solution_measure_table(_solutions)
        _solutions.solutions_to_dataframe(filtering="off", splitparams=True)
        return _solutions
