from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import vrtool.orm.models as orm_models
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.io.importers.dike_section_importer import DikeSectionImporter
from vrtool.orm.io.importers.measures.measure_result_importer import (
    MeasureResultImporter,
)
from vrtool.orm.orm_controllers import open_database

vrtool_db_default_name = "vrtool_input.db"


@dataclass
class AcceptanceTestCase:
    case_name: str
    model_directory: Path
    traject_name: str
    excluded_mechanisms: list[str] = field(
        default_factory=lambda: ["HydraulicStructures"]
    )

    @staticmethod
    def get_cases() -> list[AcceptanceTestCase]:
        # Defining acceptance test cases so they are accessible from the other test classes.
        return [
            AcceptanceTestCase(
                model_directory="TestCase1_38-1_no_housing",
                traject_name="38-1",
                excluded_mechanisms=["Revetment", "HydraulicStructures"],
                case_name="Traject 38-1, no housing",
            ),
            AcceptanceTestCase(
                model_directory="TestCase1_38-1_no_housing_stix",
                traject_name="38-1",
                excluded_mechanisms=["Revetment", "HydraulicStructures"],
                case_name="Traject 38-1, no housing, with dstability",
            ),
            AcceptanceTestCase(
                model_directory="TestCase2_38-1_overflow_no_housing",
                traject_name="38-1",
                excluded_mechanisms=["Revetment", "HydraulicStructures"],
                case_name="Traject 38-1, no-housing, with overflow",
            ),
            AcceptanceTestCase(
                model_directory="TestCase1_38-1_revetment",
                traject_name="38-1",
                excluded_mechanisms=["HydraulicStructures"],
                case_name="Traject 38-1, with revetment, case 1",
            ),
            AcceptanceTestCase(
                model_directory="TestCase3_38-1_revetment",
                traject_name="38-1",
                excluded_mechanisms=["HydraulicStructures"],
                case_name="Traject 38-1, with revetment, including bundling",
            ),
            AcceptanceTestCase(
                model_directory="TestCase4_38-1_revetment_small",
                traject_name="38-1",
                excluded_mechanisms=["HydraulicStructures"],
                case_name="Traject 38-1, two sections with revetment",
            ),
            AcceptanceTestCase(
                model_directory="TestCase3_38-1_small",
                traject_name="38-1",
                excluded_mechanisms=["Revetment", "HydraulicStructures"],
                case_name="Traject 38-1, two sections",
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
        assert not any(orm_models.AssessmentMechanismResult.select())
        assert not any(orm_models.AssessmentSectionResult.select())
        if not _connected_db.is_closed():
            _connected_db.close()

    def validate_results(self, valid_vrtool_config: VrtoolConfig):
        # Get database paths.
        _reference_database_path = valid_vrtool_config.input_database_path.with_name(
            vrtool_db_default_name
        )
        assert (
            _reference_database_path != valid_vrtool_config.input_database_path
        ), "Reference and result database point to the same Path {}.".path(
            valid_vrtool_config.input_database_path
        )

        def load_assessment_reliabilities(vrtool_db: Path) -> dict[str, pd.DataFrame]:
            _connected_db = open_database(vrtool_db)
            _assessment_reliabilities = dict(
                (_sd, DikeSectionImporter.import_assessment_reliability_df(_sd))
                for _sd in orm_models.SectionData.select()
                .join(orm_models.DikeTrajectInfo)
                .where(
                    orm_models.SectionData.dike_traject.traject_name
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
        assert not any(orm_models.MeasureResult.select())
        assert not any(orm_models.MeasureResultParameter.select())

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

        # Get database paths.
        _reference_database_path = valid_vrtool_config.input_database_path.with_name(
            vrtool_db_default_name
        )
        assert (
            _reference_database_path != valid_vrtool_config.input_database_path
        ), "Reference and result database point to the same Path {}.".path(
            valid_vrtool_config.input_database_path
        )

        def load_measures_reliabilities(
            vrtool_db: Path,
        ) -> dict[str, dict[tuple, pd.DataFrame]]:
            _connected_db = open_database(vrtool_db)
            _m_reliabilities = defaultdict(dict)
            for _measure_result in orm_models.MeasureResult.select():
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
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)

        assert any(orm_models.MeasureResult.select())
        assert not any(orm_models.OptimizationRun)
        assert not any(orm_models.OptimizationSelectedMeasure)
        assert not any(orm_models.OptimizationStep)
        assert not any(orm_models.OptimizationStepResultMechanism)
        assert not any(orm_models.OptimizationStepResultSection)

        _connected_db.close()

    def get_test_measure_result_ids(
        self, valid_vrtool_config: VrtoolConfig
    ) -> list[int]:
        _connected_db = open_database(valid_vrtool_config.input_database_path)
        _id_list = [mr.get_id() for mr in orm_models.MeasureResult.select()]
        _connected_db.close()
        return _id_list

    def validate_results(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)
        # For now just check that there are outputs.
        assert any(orm_models.OptimizationRun.select())
        assert any(orm_models.OptimizationSelectedMeasure.select())
        assert any(orm_models.OptimizationStep.select())
        assert any(orm_models.OptimizationStepResultSection.select())
        assert any(orm_models.OptimizationStepResultMechanism.select())
        _connected_db.close()


class RunFullValidator:
    def validate_acceptance_result_cases(
        self, test_results_dir: Path, test_reference_dir: Path
    ):
        files_to_compare = [
            "TakenMeasures_Doorsnede-eisen.csv",
            "TakenMeasures_Veiligheidsrendement.csv",
            "TotalCostValues_Greedy.csv",
        ]
        comparison_errors = []
        for file in files_to_compare:
            reference = pd.read_csv(
                test_reference_dir.joinpath("results", file), index_col=0
            )
            result = pd.read_csv(test_results_dir / file, index_col=0)
            try:
                assert_frame_equal(reference, result, atol=1e-6, rtol=1e-6)
            except Exception:
                comparison_errors.append("{} is different.".format(file))
        # assert no error message has been registered, else print messages
        assert not comparison_errors, "errors occured:\n{}".format(
            "\n".join(comparison_errors)
        )
