from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import vrtool.orm.models as orm
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.io.importers.dike_section_importer import DikeSectionImporter
from vrtool.orm.io.importers.measures.measure_result_importer import (
    MeasureResultImporter,
)
from vrtool.orm.orm_controllers import open_database

OptimizationStepResult = (
    orm.OptimizationStepResultMechanism | orm.OptimizationStepResultSection
)

vrtool_db_default_name = "vrtool_input.db"


def _get_database_reference_path(
    vrtool_config: VrtoolConfig, suffix_name: str = ""
) -> Path:
    # Get database paths.
    _reference_database_path = vrtool_config.input_database_path.with_name(
        f"vrtool_input{suffix_name}.db"
    )
    assert (
        _reference_database_path != vrtool_config.input_database_path
    ), "Reference and result database point to the same Path {}.".path(
        vrtool_config.input_database_path
    )
    return _reference_database_path


@dataclass
class AcceptanceTestCase:
    case_name: str
    model_directory: Path
    traject_name: str
    excluded_mechanisms: list[MechanismEnum] = field(
        default_factory=lambda: [
            MechanismEnum.HYDRAULIC_STRUCTURES,
        ]
    )

    @staticmethod
    def get_cases() -> list[AcceptanceTestCase]:
        # Defining acceptance test cases so they are accessible from the other test classes.
        return [
            AcceptanceTestCase(
                model_directory="31-1 two coastal sections",
                traject_name="31-1",
                excluded_mechanisms=[
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 31-1, two coastal sections",
            ),
            AcceptanceTestCase(
                model_directory="38-1 two river sections",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 38-1, two river sections",
            ),
            AcceptanceTestCase(
                model_directory="31-1 base coastal case",
                traject_name="31-1",
                excluded_mechanisms=[
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 31-1, base coastal case",
            ),
            AcceptanceTestCase(
                model_directory="38-1 base river case",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.REVETMENT,
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 38-1, base river case",
            ),
            AcceptanceTestCase(
                model_directory="31-1 mixed coastal case",
                traject_name="31-1",
                excluded_mechanisms=[
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 31-1, mixed coastal case",
            ),
            AcceptanceTestCase(
                model_directory="38-1 two river sections D-Stability",
                traject_name="38-1",
                excluded_mechanisms=[
                    MechanismEnum.HYDRAULIC_STRUCTURES,
                ],
                case_name="Traject 38-1, two sections with D-Stability",
            ),
        ]


class RunStepValidator(Protocol):
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        pass

    def validate_results(self, valid_vrtool_config: VrtoolConfig):
        pass


class RunStepAssessmentValidator(RunStepValidator):
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)
        assert not any(orm.AssessmentMechanismResult.select())
        assert not any(orm.AssessmentSectionResult.select())
        if not _connected_db.is_closed():
            _connected_db.close()

    def validate_results(self, valid_vrtool_config: VrtoolConfig):
        _reference_database_path = _get_database_reference_path(valid_vrtool_config)

        def load_assessment_reliabilities(vrtool_db: Path) -> dict[str, pd.DataFrame]:
            _connected_db = open_database(vrtool_db)
            _assessment_reliabilities = dict(
                (_sd, DikeSectionImporter.import_assessment_reliability_df(_sd))
                for _sd in orm.SectionData.select()
                .join(orm.DikeTrajectInfo)
                .where(
                    orm.SectionData.dike_traject.traject_name
                    == valid_vrtool_config.traject
                )
            )
            _connected_db.close()
            return _assessment_reliabilities

        _result_assessment = load_assessment_reliabilities(
            valid_vrtool_config.input_database_path
        )
        _reference_assessment = load_assessment_reliabilities(_reference_database_path)

        assert any(
            _reference_assessment.items()
        ), "No reference assessments were loaded."
        _errors = []
        for _ref_key, _ref_dataframe in _reference_assessment.items():
            _res_dataframe = _result_assessment.get(_ref_key, pd.DataFrame())
            if _res_dataframe.empty and not _ref_dataframe.empty:
                _errors.append(
                    "Section {} has no exported reliability results.".format(_ref_key)
                )
                continue
            pd.testing.assert_frame_equal(_ref_dataframe, _res_dataframe)
        if _errors:
            pytest.fail("\n".join(_errors))


class RunStepMeasuresValidator(RunStepValidator):
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)
        assert not any(orm.MeasureResult.select())
        assert not any(orm.MeasureResultParameter.select())

        if not _connected_db.is_closed():
            _connected_db.close()

    def validate_results(self, valid_vrtool_config: VrtoolConfig):
        """
        {
            "section_id":
                "measure_id":
                    "frozenset[measure_result_with_params]": reliability
        }
        """

        _reference_database_path = _get_database_reference_path(valid_vrtool_config)

        def load_measures_reliabilities(
            vrtool_db: Path,
        ) -> dict[str, dict[tuple, pd.DataFrame]]:
            _connected_db = open_database(vrtool_db)
            _m_reliabilities = defaultdict(dict)
            for _measure_result in orm.MeasureResult.select():
                _measure_per_section = _measure_result.measure_per_section
                _reliability_df = MeasureResultImporter.import_measure_reliability_df(
                    _measure_result
                )
                _available_parameters = frozenset(
                    (mrp.name, mrp.value)
                    for mrp in _measure_result.measure_result_parameters
                )
                if (
                    _available_parameters
                    in _m_reliabilities[
                        (
                            _measure_per_section.measure.name,
                            _measure_per_section.section.section_name,
                        )
                    ].keys()
                ):
                    _keys_values = [f"{k}={v}" for k, v in _available_parameters]
                    _as_string = ", ".join(_keys_values)
                    pytest.fail(
                        "Measure reliability contains twice the same parameters {}.".format(
                            _as_string
                        )
                    )
                _m_reliabilities[
                    (
                        _measure_per_section.measure.name,
                        _measure_per_section.section.section_name,
                    )
                ][_available_parameters] = _reliability_df
            _connected_db.close()
            return _m_reliabilities

        _result_assessment = load_measures_reliabilities(
            valid_vrtool_config.input_database_path
        )
        _reference_assessment = load_measures_reliabilities(_reference_database_path)

        assert any(
            _reference_assessment.items()
        ), "No reference assessments were loaded."
        _errors = []
        for _ref_key, _ref_section_measure_dict in _reference_assessment.items():
            # Iterate over each dictionary entry,
            # which represents ALL the measure results (the values)
            # of a given `MeasurePerSection` (the key).
            _res_section_measure_dict = _result_assessment.get(_ref_key, dict())
            if not any(_res_section_measure_dict.items()):
                _errors.append(
                    "Measure {} = Section {}, have no reliability results.".format(
                        _ref_key[0], _ref_key[1]
                    )
                )
                continue
            for (
                _ref_params,
                _ref_measure_result_reliability,
            ) in _ref_section_measure_dict.items():
                # Iterate over each dictionary entry,
                # which represents the measure reliability results (the values as `pd.DataFrame`)
                # for a given set of parameters represented as `dict` (the keys)
                _res_measure_result_reliability = _res_section_measure_dict.get(
                    _ref_params, pd.DataFrame()
                )
                if _res_measure_result_reliability.empty:
                    _parameters = [f"{k}={v}" for k, v in _ref_params]
                    _parameters_as_str = ", ".join(_parameters)
                    _errors.append(
                        "Measure {} = Section {}, Parameters: {}, have no reliability results".format(
                            _ref_key[0], _ref_key[1], _parameters_as_str
                        )
                    )
                    continue
                pd.testing.assert_frame_equal(
                    _ref_measure_result_reliability, _res_measure_result_reliability
                )
        if _errors:
            pytest.fail("\n".join(_errors))


class RunStepOptimizationValidator(RunStepValidator):
    _reference_db_suffix: str

    def __init__(self, reference_db_suffix: str = "") -> None:
        self._reference_db_suffix = reference_db_suffix

    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)

        assert any(orm.MeasureResult.select())
        assert not any(orm.OptimizationRun)
        assert not any(orm.OptimizationSelectedMeasure)
        assert not any(orm.OptimizationStep)
        assert not any(orm.OptimizationStepResultMechanism)
        assert not any(orm.OptimizationStepResultSection)

        _connected_db.close()

    def _get_opt_run(self, database_path: Path, method: str) -> list[orm.OptimizationRun]:
        """
        Gets a list of all existing `OptimizationRun` rows in the database with all the backrefs related to an "optimization" already instantiated ("eager loading").

        IMPORTANT! We instantiate everything so that OptimizationRun has (in memory) access to all the backrefs.
        We could also decide to return `orm.OptimizationStepResultMechanism` and `orm.OptimizationStepResultSection`,
        however that would make it 'harder' to validate later on, so we prefer an 'uglier' loading of data for a more
        readable validation.

        Args:
            database_path (Path): Location of the `sqlite` database file.
            method (str): Name of the optimization method as given in the configuration file.

        Returns:
            list[orm.OptimizationRun]: Collection of `OptimizationRun` with insantiated backrefs.
        """
        opt_run_list = []
        with open_database(database_path):
            for opt_run in list(
                orm.OptimizationRun.select(
                    orm.OptimizationRun,
                    orm.OptimizationSelectedMeasure,
                    orm.OptimizationStep,
                    orm.OptimizationStepResultSection,
                    orm.OptimizationStepResultMechanism,
                    orm.OptimizationType
                )
                .join_from(orm.OptimizationRun, orm.OptimizationType)
                .join_from(orm.OptimizationRun, orm.OptimizationSelectedMeasure)
                .join_from(orm.OptimizationSelectedMeasure, orm.OptimizationStep)
                .join_from(
                    orm.OptimizationStep,
                    orm.OptimizationStepResultSection,
                )
                .join_from(
                    orm.OptimizationStep,
                    orm.OptimizationStepResultMechanism,
                )
                .group_by(orm.OptimizationRun)
            ):
                if opt_run.optimization_type.name == method.upper():
                    opt_run.optimization_run_measure_results = list(
                        opt_run.optimization_run_measure_results
                    )
                    for opt_selected_measure in opt_run.optimization_run_measure_results:
                        opt_selected_measure.optimization_steps = list(
                            opt_selected_measure.optimization_steps
                        )
                        for opt_step in opt_selected_measure.optimization_steps:
                            opt_step.optimization_step_results_mechanism = list(
                                opt_step.optimization_step_results_mechanism
                            )
                            opt_step.optimization_step_results_section = list(
                                opt_step.optimization_step_results_section
                            )
                    opt_run_list.append(opt_run)
        return opt_run_list

    def _compare_optimization_run(
        self,
        reference: orm.OptimizationRun,
        result: orm.OptimizationRun,
    ):
        # We set names for the runs based on their type, so we only need to check the last name.
        assert reference.name.split(" ")[-1] == result.name.split(" ")[-1]
        assert reference.discount_rate == result.discount_rate
        assert reference.optimization_type.name == result.optimization_type.name

        self._compare_optimization_selected_measures(
            reference.optimization_run_measure_results,
            result.optimization_run_measure_results,
        )

    def _compare_measure_per_section(
        self, reference: orm.MeasurePerSection, result: orm.MeasurePerSection
    ):
        assert reference.section.section_name == result.section.section_name
        assert reference.measure.name == result.measure.name

    def _compare_optimization_selected_measures(
        self,
        reference_list: list[orm.OptimizationSelectedMeasure],
        result_list: list[orm.OptimizationSelectedMeasure],
    ):
        # Check row size.
        assert len(reference_list) == len(result_list)

        # Check each row individually.
        for _idx, _reference in enumerate(reference_list):
            _result = result_list[_idx]
            assert _reference.investment_year == _result.investment_year
            self._compare_measure_per_section(
                _reference.measure_result.measure_per_section,
                _result.measure_result.measure_per_section,
            )
            self._compare_optimization_steps(
                _reference.optimization_steps,
                _result.optimization_steps,
            )

    def _compare_optimization_steps(
        self,
        reference_list: list[orm.OptimizationStep],
        result_list: list[orm.OptimizationStep],
    ):
        # Check collection size.
        assert len(reference_list) == len(result_list)

        # Check each row individually.
        for _idx, _reference in enumerate(reference_list):
            _result = result_list[_idx]
            assert _reference.step_number == _result.step_number
            if not _reference.total_lcc == None: #TODO: temporary fix as long as references dont contain cost for TR.
                assert _reference.total_lcc == pytest.approx(_result.total_lcc, abs=1e-2)
                assert _reference.total_risk == pytest.approx(_result.total_risk, abs=1e-2)
            self._compare_optimization_step_results_mechanism(
                _reference.optimization_step_results_mechanism,
                _result.optimization_step_results_mechanism,
            )
            self._compare_optimization_step_results_section(
                _reference.optimization_step_results_section,
                _result.optimization_step_results_section,
            )

    def _compare_optimization_step_result(
        self,
        reference: OptimizationStepResult,
        result: OptimizationStepResult,
    ):
        assert reference.beta == pytest.approx(result.beta, abs=1e-4)
        assert reference.time == result.time
        assert reference.lcc == pytest.approx( result.lcc, abs=1e-2)

    def _compare_mechanism_per_section(
        self, reference: orm.MechanismPerSection, result: orm.MechanismPerSection
    ):
        assert reference.section.section_name == result.section.section_name
        assert reference.mechanism.name == result.mechanism.name

    def _compare_optimization_step_results_mechanism(
        self,
        reference_list: list[orm.OptimizationStepResultMechanism],
        result_list: list[orm.OptimizationStepResultMechanism],
    ):
        # Check collection size.
        assert len(reference_list) == len(result_list)

        # Check each row individually.
        for _idx, _reference in enumerate(reference_list):
            _result = result_list[_idx]
            self._compare_optimization_step_result(_reference, _result)
            self._compare_mechanism_per_section(
                _reference.mechanism_per_section,
                _result.mechanism_per_section,
            )

    def _compare_optimization_step_results_section(
        self,
        reference_list: list[orm.OptimizationStepResultSection],
        result_list: list[orm.OptimizationStepResultSection],
    ):
        # Check collection size.
        assert len(reference_list) == len(result_list)

        # Check each row individually.
        for _idx, _reference in enumerate(reference_list):
            self._compare_optimization_step_result(_reference, result_list[_idx])

    def validate_results(self, valid_vrtool_config: VrtoolConfig):
        # Steps for validation.
        # Load optimization runs.
        _reference_path = _get_database_reference_path(valid_vrtool_config, self._reference_db_suffix)
        _run_data = {method: (self._get_opt_run(_reference_path, method), self._get_opt_run(valid_vrtool_config.input_database_path, method)) 
                     for method in valid_vrtool_config.design_methods}

        for method in _run_data.keys():
            #verify there is an equal number of runs of each type
            assert len(_run_data[method][0]) == len(_run_data[method][1])

        # Because the resulting database does not contain the previous results,
        # we can then assume all the values will be in the same exact order.
        for _runs in _run_data.values():
            [self._compare_optimization_run(_reference_run, _result_run) for _reference_run, _result_run in zip(*_runs)]

    @staticmethod
    def get_csv_reference_dir(vrtool_config: VrtoolConfig) -> Path:
        return vrtool_config.input_directory.joinpath("reference")


class RunFullValidator(RunStepValidator):
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        assert RunStepOptimizationValidator.get_csv_reference_dir(
            valid_vrtool_config
        ).exists()

    def validate_results(self, valid_vrtool_config: VrtoolConfig):
        # Validate the optimization results.
        # TODO: Remove this validator class if we understand that
        # the optimization validation is enough.
        _optimization_validator = RunStepOptimizationValidator()
        _optimization_validator.validate_results(valid_vrtool_config)
