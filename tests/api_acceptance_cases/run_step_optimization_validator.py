from __future__ import annotations

from pathlib import Path

import pytest

import vrtool.orm.models as orm
from tests.api_acceptance_cases.run_step_validator_protocol import (
    RunStepValidator,
    _get_database_reference_path,
)
from tests.postprocessing_report import PostProcessingReport
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.orm_controllers import open_database

OptimizationStepResult = (
    orm.OptimizationStepResultMechanism | orm.OptimizationStepResultSection
)


class RunStepOptimizationValidator(RunStepValidator):
    _reference_db_suffix: str
    _report_validation: bool

    def __init__(
        self, reference_db_suffix: str = "", report_validation: bool = True
    ) -> None:
        self._reference_db_suffix = reference_db_suffix
        self._report_validation = report_validation

    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)

        assert any(orm.MeasureResult.select())
        assert not any(orm.OptimizationRun)
        assert not any(orm.OptimizationSelectedMeasure)
        assert not any(orm.OptimizationStep)
        assert not any(orm.OptimizationStepResultMechanism)
        assert not any(orm.OptimizationStepResultSection)

        _connected_db.close()

    def _get_opt_run(
        self, database_path: Path, method: str
    ) -> list[orm.OptimizationRun]:
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
                    orm.OptimizationType,
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
                    for (
                        opt_selected_measure
                    ) in opt_run.optimization_run_measure_results:
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
            assert (
                _reference.step_number == _result.step_number
            ), f"Expected {_reference.step_number} but got {_result.step_number}"
            if (
                _reference.total_lcc is not None
            ):  # TODO: temporary fix as long as references don't contain cost for TR.
                assert _reference.total_lcc == pytest.approx(
                    _result.total_lcc, abs=1e-2
                )
                assert _reference.total_risk == pytest.approx(
                    _result.total_risk, abs=1e-2
                )
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
        assert reference.lcc == pytest.approx(result.lcc, abs=1e-2)

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

    def _generate_postprocessing_report(
        self, reference_path: Path, results_path: Path
    ) -> None:
        with PostProcessingReport(
            reference_db=reference_path,
            result_db=results_path,
            report_dir=results_path.parent.joinpath(
                "postprocessing_report_" + results_path.stem
            ),
        ) as _pp_report:
            _found_errors = _pp_report.generate_report()

        if not self._report_validation:
            return

        # TODO: Due to time concerns we reuse the logic from the report
        # to compare optimization lcc with measure result cost by simply
        # reading the 'lines' in their respective reports.

        if any(_found_errors.values()):
            _error_header = "For {}, the following errors were found: \n"
            _error_str = ""
            for _opt_run, _errors in _found_errors.items():
                _error_str += _error_header.format(_opt_run)
                _error_str += "\n\t".join(_errors)
                _error_str += "\n"

            pytest.fail(_error_str)

    def validate_results(self, valid_vrtool_config: VrtoolConfig):
        # Steps for validation.
        _reference_path = _get_database_reference_path(
            valid_vrtool_config, self._reference_db_suffix
        )

        # Generate postprocessing report
        self._generate_postprocessing_report(
            _reference_path, valid_vrtool_config.input_database_path
        )

        # Load optimization runs.
        _run_data = {
            method: (
                self._get_opt_run(_reference_path, method),
                self._get_opt_run(valid_vrtool_config.input_database_path, method),
            )
            for method in valid_vrtool_config.design_methods
        }

        for method in _run_data.keys():
            # verify there is an equal number of runs of each type
            assert len(_run_data[method][0]) == len(_run_data[method][1])

        # Because the resulting database does not contain the previous results,
        # we can then assume all the values will be in the same exact order.
        for _runs in _run_data.values():
            for _reference_run, _result_run in zip(*_runs):
                self._compare_optimization_run(_reference_run, _result_run)

    @staticmethod
    def get_csv_reference_dir(vrtool_config: VrtoolConfig) -> Path:
        return vrtool_config.input_directory.joinpath("reference")
