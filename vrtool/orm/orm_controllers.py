import logging
from pathlib import Path

from peewee import SqliteDatabase

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.orm import models as orm
from vrtool.orm.io.exporters.measures.solutions_exporter import SolutionsExporter
from vrtool.orm.io.exporters.optimization.strategy_base_exporter import (
    StrategyBaseExporter,
)
from vrtool.orm.io.exporters.safety_assessment.dike_section_reliability_exporter import (
    DikeSectionReliabilityExporter,
)
from vrtool.orm.io.importers.dike_traject_importer import DikeTrajectImporter
from vrtool.orm.io.importers.measures.solutions_for_measure_results_importer import (
    SolutionsForMeasureResultsImporter,
)
from vrtool.orm.io.importers.measures.solutions_importer import (
    SolutionsImporter,
)
from vrtool.orm.orm_db import vrtool_db
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)

import itertools


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
    logging.info("Opened connection to export Dike's section reliability.")
    _exporter = DikeSectionReliabilityExporter()
    for _section in result.selected_traject.sections:
        _exporter.export_dom(_section)
    _connected_db.close()
    logging.info("Closed connection after export for Dike's section reliability.")


def clear_assessment_results(config: VrtoolConfig) -> None:
    """
    Clears all the assessment results from the database

    Args:
        config (VrtoolConfig): Vrtool configuration
    """

    open_database(config.input_database_path)
    logging.info("Opened connection for clearing initial assessment results.")

    with vrtool_db.atomic():
        orm.AssessmentMechanismResult.delete().execute()
        orm.AssessmentSectionResult.delete().execute()

    vrtool_db.close()

    logging.info("Closed connection after clearing initial assessment results.")


def clear_measure_results(config: VrtoolConfig) -> None:
    """
    Clears all the measure results from the database

    Args:
        config (VrtoolConfig): Vrtool configuration
    """

    open_database(config.input_database_path)
    logging.info("Opened connection for clearing measure results.")

    with vrtool_db.atomic():
        orm.MeasureResult.delete().execute()
        # This table should be cleared 'on cascade'.
        orm.MeasureResultParameter.delete().execute()
        orm.MeasureResultSection.delete().execute()
        orm.MeasureResultMechanism.delete().execute()

    vrtool_db.close()

    logging.info("Closed connection after clearing measure results.")


def clear_optimization_results(config: VrtoolConfig) -> None:
    """
    Clears all the optimization related results from the database.

    Args:
        config (VrtoolConfig): Vrtool configuration.
    """
    open_database(config.input_database_path)
    logging.info("Opened connection for clearing optimization results.")

    with vrtool_db.atomic():
        orm.OptimizationRun.delete().execute()
        # These tables should be cleared 'on cascade'.
        orm.OptimizationSelectedMeasure.delete().execute()
        orm.OptimizationStep.delete().execute()
        orm.OptimizationStepResult.delete().execute()

    vrtool_db.close()

    logging.info("Closed connection after clearing optimization results.")


def export_results_measures(result: ResultsMeasures) -> None:
    """
    Exports the solutions to a database

    Args:
        result (ResultsMeasures): result of measure step
    """

    _connected_db = open_database(result.vr_config.input_database_path)

    logging.info("Opened connection to export solution.")

    _exporter = SolutionsExporter()
    for _solution in result.solutions_dict.values():
        _exporter.export_dom(_solution)
    _connected_db.close()

    logging.info("Closed connection after export solution.")


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
    config: VrtoolConfig, results_ids_to_import: list[int]
) -> ResultsMeasures:
    """
    Imports results masures from a database into a `ResultsMeasure` instance.

    Args:
        config (VrtoolConfig): Configuration containing database path.
        results_ids_to_import (list[int]): List of measure results' IDs.

    Returns:
        ResultsMeasures: Instance hosting all the required measures' results.
    """
    _dike_traject = get_dike_traject(config)
    open_database(config.input_database_path)

    _solutions_dict = dict()
    # Group the measure results by section.
    measure_results = orm.MeasureResult.select().where(orm.MeasureResult.id.in_(results_ids_to_import))

    _grouped_by_section = [
        (_section, list(_grouped_measure_results))
        for _section, _grouped_measure_results in itertools.groupby(
            measure_results, lambda x: x.measure_per_section.section
        )
    ]

    # Import a solution per section:
    for _section, _selected_measure_results in _grouped_by_section:
        # Import measures into solution
        _mapped_section = next(
            _ds for _ds in _dike_traject.sections if _ds.name == _section.section_name
        )
        _imported_solution = SolutionsForMeasureResultsImporter(
            config,
            _mapped_section,
        ).import_orm(_selected_measure_results)
        _solutions_dict[_section.section_name] = _imported_solution
    _dike_traject.set_probabilities()
    vrtool_db.close()

    _results_measures = ResultsMeasures()
    _results_measures.solutions_dict = _solutions_dict
    _results_measures.selected_traject = _dike_traject
    _results_measures.vr_config = config
    _results_measures.ids_to_import = results_ids_to_import

    return _results_measures


def create_optimization_run(
    vr_config: VrtoolConfig,
    selected_measure_result_ids: list[int],
    optimization_name: str,
) -> None:
    """
    Creates an `OptimizationRun` database entry and as many entries as needed
    in the `OptimizationSelectedMeasure` table based on the provided arguments.

    Args:
        vr_config (VrtoolConfig): Configuration containing optimization methods and discount rate to be used.
        selected_measure_result_ids (list[int]): list of `MeasureResult` id's in the database.
        optimization_name (str): name to give to an optimization run.
    """
    _connected_db = open_database(vr_config.input_database_path)
    logging.info(
        "Opened connection to export optimization run {}.".format(optimization_name)
    )
    for _method_type in vr_config.design_methods:
        _optimization_type, _ = orm.OptimizationType.get_or_create(
            name=_method_type.upper()
        )
        _optimization_run = orm.OptimizationRun.create(
            name=optimization_name,
            discount_rate=vr_config.discount_rate,
            optimization_type=_optimization_type,
        )
        orm.OptimizationSelectedMeasure.insert_many(
            [
                dict(
                    optimization_run=_optimization_run,
                    measure_result=orm.MeasureResult.get_by_id(_measure_id),
                    investment_year=0,
                )
                for _measure_id in selected_measure_result_ids
            ]
        ).execute()

    logging.info(
        "Closed connection after export optimization run {}.".format(optimization_name)
    )
    _connected_db.close()


def export_results_optimization(result: ResultsOptimization) -> None:
    """
    Exports the optimization results (`list[StrategyBase]`) to a database.

    Args:
        result (ResultsOptimization): result of an optimization run.
    """

    _connected_db = open_database(result.vr_config.input_database_path)

    logging.info("Opened connection to export optimizations.")

    _exporter = StrategyBaseExporter()
    for _strategy_result in result.results_strategies:
        _exporter.export_dom(_strategy_result)
    _connected_db.close()

    logging.info("Closed connection after export optimizations.")
