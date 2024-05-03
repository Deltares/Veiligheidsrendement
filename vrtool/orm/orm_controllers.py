import itertools
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd
from peewee import SqliteDatabase
from tqdm import tqdm

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.orm import models as orm
from vrtool.orm.io.exporters.measures.dict_to_custom_measure_exporter import (
    DictListToCustomMeasureExporter,
)
from vrtool.orm.io.exporters.measures.solutions_exporter import SolutionsExporter
from vrtool.orm.io.exporters.optimization.strategy_exporter import StrategyExporter
from vrtool.orm.io.exporters.safety_assessment.dike_section_reliability_exporter import (
    DikeSectionReliabilityExporter,
)
from vrtool.orm.io.importers.dike_traject_importer import DikeTrajectImporter
from vrtool.orm.io.importers.measures.solutions_for_measure_results_importer import (
    SolutionsForMeasureResultsImporter,
)
from vrtool.orm.io.importers.measures.solutions_importer import SolutionsImporter
from vrtool.orm.io.importers.optimization.optimization_section_as_input_importer import (
    OptimizationSectionAsInputImporter,
)
from vrtool.orm.io.importers.optimization.optimization_step_importer import (
    OptimizationStepImporter,
)
from vrtool.orm.orm_db import vrtool_db
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)


def initialize_database(database_path: Path) -> SqliteDatabase:
    """
    Generates an empty SQLite database with all the tables requried by the `Vrtool`.

    Args:
        database_path (Path): Location where to save the database.

    Returns:
        SqliteDatabase: The initialized instance of the database.
    """
    if not database_path.parent.exists():
        database_path.parent.mkdir(parents=True)

    vrtool_db.init(database_path)
    vrtool_db.connect()
    vrtool_db.create_tables(
        [
            orm.SectionData,
            orm.Buildings,
            orm.Mechanism,
            orm.MechanismPerSection,
            orm.ComputationType,
            orm.ComputationScenario,
            orm.ComputationScenarioParameter,
            orm.MechanismTable,
            orm.CharacteristicPointType,
            orm.ProfilePoint,
            orm.WaterlevelData,
            orm.MeasureType,
            orm.CombinableType,
            orm.Measure,
            orm.StandardMeasure,
            orm.CustomMeasure,
            orm.DikeTrajectInfo,
            orm.SupportingFile,
            orm.MeasurePerSection,
            orm.CustomMeasureParameter,
            orm.SlopePart,
            orm.BlockRevetmentRelation,
            orm.GrassRevetmentRelation,
            orm.AssessmentSectionResult,
            orm.AssessmentMechanismResult,
        ]
        + orm.get_optimization_results_tables()
        + orm.get_measure_results_tables()
    )
    return vrtool_db


def open_database(database_path: Path) -> SqliteDatabase:
    """
    Initializes and connects the `Vrtool` to its related database.

    Args:
        database_path (Path): Location of the SQLite database.

    Returns:
        SqliteDatabase: Initialized database.
    """
    if not database_path.exists():
        raise ValueError("No file was found at {}".format(database_path))
    vrtool_db.init(database_path)
    vrtool_db.connect()
    return vrtool_db


def get_dike_traject(config: VrtoolConfig) -> DikeTraject:
    """
    Returns a dike traject with all the required section data.

    Args:
        config (VrtoolConfig): Configuration object model containing the traject's information as well as the database's location path.
    """
    open_database(config.input_database_path)
    _dike_traject = DikeTrajectImporter(config).import_orm(
        orm.DikeTrajectInfo.get(orm.DikeTrajectInfo.traject_name == config.traject)
    )
    vrtool_db.close()
    return _dike_traject


def get_dike_section_solutions(
    config: VrtoolConfig, dike_section: DikeSection, general_info: DikeTrajectInfo
) -> Solutions:
    """
    Gets the `solutions` instance with all measures (`MeasureBase`) mapped to the orm `Measure` table.

    Args:
        config (VrtoolConfig): Vrtool configuration.
        dike_section (DikeSection): Selected DikeSection whose measures need to be loaded.
        general_info (DikeTrajectInfo): Required data structure to evaluate measures after import.

    Returns:
        Solutions: instance with all related measures (standard and / or custom).
    """
    open_database(config.input_database_path)
    _importer = SolutionsImporter(config, dike_section)
    _orm_section_data = orm.SectionData.get(
        orm.SectionData.section_name == dike_section.name
    )
    _solutions = _importer.import_orm(_orm_section_data)
    vrtool_db.close()
    _solutions.evaluate_solutions(dike_section, general_info, preserve_slope=False)
    return _solutions


def export_results_safety_assessment(result: ResultsSafetyAssessment) -> None:
    """
    Exports the (initial) safety assessments results in a `ResultsSafetyAssessment` instance to the database defined in its `VrtoolConfig` field.
    The database connection will be opened and closed within the call to this method.

    Args:
        result (ResultsSafetyAssessment): Instance containing dike sections' reliability and output database's location.
    """
    _connected_db = open_database(result.vr_config.input_database_path)
    logging.debug("Opened connection to export Dike's section reliability.")
    _exporter = DikeSectionReliabilityExporter()
    for _section in result.selected_traject.sections:
        _exporter.export_dom(_section)
    _connected_db.close()
    logging.info("Resultaten beoordeling & projectie geexporteerd naar database.")


def clear_assessment_results(config: VrtoolConfig) -> None:
    """
    Clears all the assessment results from the database

    Args:
        config (VrtoolConfig): Vrtool configuration
    """

    open_database(config.input_database_path)
    logging.debug("Opened connection for clearing initial assessment results.")

    with vrtool_db.atomic():
        orm.AssessmentMechanismResult.delete().execute(vrtool_db)
        orm.AssessmentSectionResult.delete().execute(vrtool_db)

    vrtool_db.close()

    logging.info("Bestaande beoordelingsresultaten verwijderd.")


def clear_measure_results(config: VrtoolConfig) -> None:
    """
    Clears all the measure results from the database

    Args:
        config (VrtoolConfig): Vrtool configuration
    """

    open_database(config.input_database_path)
    logging.debug("Opened connection for clearing measure results.")

    with vrtool_db.atomic():
        orm.MeasureResult.delete().execute(vrtool_db)
        # This table should be cleared 'on cascade'.
        orm.MeasureResultParameter.delete().execute(vrtool_db)
        orm.MeasureResultSection.delete().execute(vrtool_db)
        orm.MeasureResultMechanism.delete().execute(vrtool_db)

    vrtool_db.close()

    logging.info("Bestaande resultaten voor maatregelen verwijderd.")


def clear_optimization_results(config: VrtoolConfig) -> None:
    """
    Clears all the optimization related results from the database.

    Args:
        config (VrtoolConfig): Vrtool configuration.
    """
    open_database(config.input_database_path)
    logging.debug("Opened connection for clearing optimization results.")

    with vrtool_db.atomic():
        orm.OptimizationRun.delete().execute(vrtool_db)
        # These tables should be cleared 'on cascade'.
        orm.OptimizationSelectedMeasure.delete().execute(vrtool_db)
        orm.OptimizationStep.delete().execute(vrtool_db)
        orm.OptimizationStepResultMechanism.delete().execute(vrtool_db)
        orm.OptimizationStepResultSection.delete().execute(vrtool_db)

    vrtool_db.close()

    logging.info("Bestaande optimalisatieresultaten verwijderd.")


def export_results_measures(result: ResultsMeasures) -> None:
    """
    Exports the solutions to a database

    Args:
        result (ResultsMeasures): result of measure step
    """

    _connected_db = open_database(result.vr_config.input_database_path)

    logging.info("Start export resultaten maatregelen naar database.")

    _exporter = SolutionsExporter()
    for _solution in tqdm(
        result.solutions_dict.values(),
        desc="Aantal geexporteerde dijkvakken:",
        total=len(result.solutions_dict),
        unit="vak",
    ):
        _exporter.export_dom(_solution)
    _connected_db.close()

    logging.debug("Export van resultaten maatregelen afgerond.")


def get_exported_measure_result_ids(result_measures: ResultsMeasures) -> list[int]:
    """
    Retrieves from the database the list of IDs for the provided results measures.
    To do so we check all the available `MeasureResult` related to the `MeasurePerSection`
    contained in the `ResultsMeasures` object.

    Args:
        result_measures (ResultsMeasures): Result measures' whose ids need to be retrieved.

    Returns:
        list[int]: List of IDs of `MeasureResult`.
    """
    _connected_db = open_database(result_measures.vr_config.input_database_path)
    _result_measure_ids = []
    for _solution in result_measures.solutions_dict.values():
        for _measure in _solution.measures:
            _measure_per_section = SolutionsExporter.get_measure_per_section(
                _solution.section_name,
                _solution.config.traject,
                _measure.parameters["ID"],
            )
            _result_measure_ids.extend(
                [
                    mxsr.get_id()
                    for mxsr in _measure_per_section.measure_per_section_result
                ]
            )

    _connected_db.close()
    return _result_measure_ids


def import_results_measures(
    config: VrtoolConfig, results_ids_to_import: list[tuple[int, int]]
) -> ResultsMeasures:
    """
    Imports results masures from a database into a `ResultsMeasure` instance.

    Args:
        config (VrtoolConfig): Configuration containing database path.
        results_ids_to_import (list[tuple[int, int]]): List of measure results' IDs. including their respective investment year.

    Returns:
        ResultsMeasures: Instance hosting all the required measures' results.
    """
    _dike_traject = get_dike_traject(config)
    open_database(config.input_database_path)

    _solutions_dict = dict()

    _mr_list = [
        (orm.MeasureResult.get_by_id(_result_tuple[0]), _result_tuple[1])
        for _result_tuple in results_ids_to_import
    ]
    _grouped_by_section = [
        (_section, list(_grouped_measure_results))
        for _section, _grouped_measure_results in itertools.groupby(
            _mr_list, lambda x: x[0].measure_per_section.section
        )
    ]
    # If there are non_unique sections, we will have to combine the results.
    # get the first entries of the tuples in _grouped_by_section
    _grouped_by_section_unique = defaultdict(list)
    for key, value in _grouped_by_section:
        _grouped_by_section_unique[key].extend(value)
    _grouped_by_section_unique = list(_grouped_by_section_unique.items())

    # Import a solution per section:
    for _section, _selected_measure_year_results in _grouped_by_section_unique:
        _selected_measure_id, _selected_measure_year = zip(
            *_selected_measure_year_results
        )
        # Import measures into solution
        _mapped_section = next(
            _ds for _ds in _dike_traject.sections if _ds.name == _section.section_name
        )
        _imported_solution = SolutionsForMeasureResultsImporter(
            config,
            _mapped_section,
        ).import_orm(_selected_measure_id)
        _solutions_dict[_section.section_name] = _imported_solution
        # NOTE: This is an "implicit" filter of which years will be imported
        # based on the input "investment_year".
        _solutions_dict[_section.section_name].MeasureData[
            "year"
        ] = _selected_measure_year
    _dike_traject.set_probabilities()
    vrtool_db.close()

    _results_measures = ResultsMeasures()
    _results_measures.solutions_dict = _solutions_dict
    _results_measures.selected_traject = _dike_traject
    _results_measures.vr_config = config
    _results_measures.ids_to_import = results_ids_to_import

    return _results_measures


def import_results_measures_for_optimization(
    config: VrtoolConfig, results_ids_to_import: list[tuple[int, int]]
) -> list[SectionAsInput]:
    """
    Method used to import all requested measure results for an Optimization run.
    It is meant to deprecate / replace / remove the previous `import_result_measures`.

    Args:
        config (VrtoolConfig): Vrtool configuration containing connection and cost details.
        results_ids_to_import (list[tuple[int, int]]): List of measure results' IDs. including their respective investment year.

    Returns:
        list[SectionAsInput]: Mapped sections with relevant measure results data.
    """

    def get_measure_results_to_import() -> (
        dict[orm.SectionData, dict[orm.MeasureResult, list[int]]]
    ):
        """
        Returns a dictionary of `orm.SectionData` containing dictionaries of their
        to-be-imported `orm.MeasureResult` with their respective `investment_year`.
        """
        _section_measure_result_dict = defaultdict(lambda: defaultdict(list))
        for _result_tuple in results_ids_to_import:
            _measure_result = orm.MeasureResult.get_by_id(_result_tuple[0])
            _measure_section = _measure_result.measure_per_section.section
            _section_measure_result_dict[_measure_section][_measure_result].append(
                _result_tuple[1]
            )

        return _section_measure_result_dict

    # Import a solution per section:
    _list_section_as_input: list[SectionAsInput] = []
    with open_database(config.input_database_path).connection_context():
        _importer = OptimizationSectionAsInputImporter(config)
        _list_section_as_input = list(
            map(
                _importer.import_from_section_data_results,
                get_measure_results_to_import().items(),
            )
        )

    return _list_section_as_input


def get_all_measure_results_with_supported_investment_years(
    valid_vrtool_config: VrtoolConfig,
) -> list[tuple[int, int]]:
    """
    Gets all available measure results (`MeasureResult`) from the database paired
    to a valid investment year (only year 0).

    Args:
        valid_vrtool_config (VrtoolConfig):
            Configuration contanining database connection details.

    Returns:
        list[tuple[int, int]]: List of measure result - investment year pairs.
    """
    _connected_db = open_database(valid_vrtool_config.input_database_path)
    # We do not want measures that have a year variable >0 initially, as then the interpolation is messed up.
    _supported_measures = (
        orm.MeasureResult.select()
        .join(orm.MeasurePerSection)
        .join(orm.Measure)
        .where(orm.Measure.year == 0)
    )
    _connected_db.close()

    _measure_result_with_year_list = []
    for _measure_result in _supported_measures:
        # All will get at least year 0.
        _measure_result_with_year_list.append((_measure_result.get_id(), 0))
        if _measure_result.measure_per_section.measure.measure_type.name in (
            MeasureTypeEnum.SOIL_REINFORCEMENT.get_old_name(),
            MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN.get_old_name(),
        ):
            # For those of type "Soil reinforcement" we also add year 20.
            _measure_result_with_year_list.append((_measure_result.get_id(), 20))

    return _measure_result_with_year_list


def _normalize_optimization_run_name(
    optimization_name: str, optimization_type: str
) -> str:
    return f"{optimization_name} {optimization_type}"


def create_optimization_run_for_selected_measures(
    vr_config: VrtoolConfig,
    optimization_name: str,
    selected_measure_results_year: list[tuple[int, int]],
) -> dict[int, list[int]]:
    """
    Imports all the selected `MeasureResult` entries and creates an `OptimizationRun`
    database entry and as many entries as needed in the `OptimizationSelectedMeasure`
    table based on the provided arguments.

    This is the method to call for running 'lose' single optimization runs.

    Args:
        vr_config (VrtoolConfig): Configuration containing optimization methods and discount rate to be used.
        optimization_name (str): name to give to an optimization run.
        selected_measure_results_year (list[tuple[int, int]]): list of `MeasureResult` id's in the database including their respective investment year.

    Returns:
        dict[int, list[int]: A dictionary mapping each selected measure to an optimization run.
    """

    with open_database(vr_config.input_database_path) as _db:
        logging.debug(
            "Opened connection to export optimization run {}.".format(optimization_name)
        )
        _optimization_selected_measure_ids = defaultdict(list)
        for _method_type in vr_config.design_methods:
            _optimization_type, _ = orm.OptimizationType.get_or_create(
                name=_method_type.upper()
            )
            _optimization_run: orm.OptimizationRun = orm.OptimizationRun.create(
                name=_normalize_optimization_run_name(optimization_name, _method_type),
                discount_rate=vr_config.discount_rate,
                optimization_type=_optimization_type,
            )
            orm.OptimizationSelectedMeasure.insert_many(
                [
                    dict(
                        optimization_run=_optimization_run,
                        measure_result=orm.MeasureResult.get_by_id(_measure_id[0]),
                        investment_year=_measure_id[1],
                    )
                    for _measure_id in selected_measure_results_year
                ]
            ).execute(_db)
            # from orm.OptimizationSelectedMeasure get all ids where optimization_run_id = _optimization_run.id
            _optimization_selected_measure_ids[_optimization_run.id] = list(
                map(lambda x: x.id, _optimization_run.optimization_run_measure_results)
            )

    logging.info(
        "Closed connection after export optimization run %s.", optimization_name
    )

    return _optimization_selected_measure_ids


def export_results_optimization(
    result: ResultsOptimization, run_ids: list[int]
) -> None:
    """
    Exports the optimization results (`list[StrategyProtocol]`) to a database.

    Args:
        result (ResultsOptimization): result of an optimization run.
    """

    _connected_db = open_database(result.vr_config.input_database_path)

    logging.debug("Opened connection to export optimizations.")

    for _run_id, _result_strategy in zip(run_ids, result.results_strategies):
        _exporter = StrategyExporter(_run_id)
        _exporter.export_dom(_result_strategy)
    _connected_db.close()

    logging.info("Resultaten geexporteerd.")


def get_optimization_steps(optimization_run_id: int) -> Iterator[orm.OptimizationStep]:
    """
    DISCLAIMER: An open database connection is required to run this call!
    """
    _optimization_run = orm.OptimizationRun.get_by_id(optimization_run_id)
    return (
        orm.OptimizationStep.select()
        .join(orm.OptimizationSelectedMeasure)
        .where(
            orm.OptimizationStep.optimization_selected_measure.optimization_run
            == _optimization_run
        )
    )


def get_optimization_step_with_lowest_total_cost(
    vrtool_config: VrtoolConfig, optimization_run_id: int
) -> tuple[orm.OptimizationStep, pd.DataFrame, float]:
    """
    Gets the `OptimizationStep` with the lowest *total* cost.
    The total cost is calculated based on `LCC` and risk.

    Args:
        vrtool_db_path (Path): Sqlite database path.

    Returns:
        orm.OptimizationStep: The `OptimizationStep` instance with the lowest *total* cost
    """
    _connected_db = open_database(vrtool_config.input_database_path)
    logging.debug(
        "Openned connection to retrieve 'OptimizationStep' with lowest total cost."
    )

    _results = []
    for _optimization_step in get_optimization_steps(optimization_run_id):
        _as_df = OptimizationStepImporter.import_optimization_step_results_df(
            _optimization_step
        )
        _cost = _optimization_step.total_lcc + _optimization_step.total_risk
        _results.append((_optimization_step, _as_df, _cost))

    _connected_db.close()
    logging.debug(
        "Closed connection after retrieval of lowest total cost 'OptimizationStep'."
    )

    return min(_results, key=lambda results_tuple: results_tuple[2])


def add_custom_measures(
    vrtool_config: VrtoolConfig, custom_measures: list[dict]
) -> list[orm.CustomMeasure]:
    """
    Maps the provided list of dictionaries, ( with keys `MEASURE_NAME`,
    `COMBINABLE_TYPE`, `SECTION_NAME`, `MECHANISM_NAME`, `TIME`,
     `COST`, `BETA`), into a `CustomMeasure` and all the related
    (required) other tables such as `Measure` or `MeasureResult`.

    Args:
        vrtool_config (VrtoolConfig): Configuration to be used for this tool.
        custom_measures (list[dict]): List of dictionaries, each one representing a
        `CustomMeasure`.

    Returns:
        list[orm.CustomMeasure]: list with id's of the created custom measures.
    """

    # 1. The list of dictionaries should be grouped by the `MEASURE_NAME` key.
    # We assume that all custom measures with the same name also have the same
    # `COMBINABLE_TYPE` and `TIME`
    _exported_measures = []

    with open_database(vrtool_config.input_database_path) as _db:
        _exported_measures = DictListToCustomMeasureExporter(
            _db, vrtool_config
        ).export_dom(custom_measures)

    # 4. Return the list of generated custom measures.
    # (This step could be replaced with returning a new dataclass type.)
    return _exported_measures
