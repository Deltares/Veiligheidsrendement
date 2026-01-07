from __future__ import annotations

from pathlib import Path

import pytest

import vrtool.orm.models as orm
from tests.api_acceptance_cases.run_step_validator_protocol import (
    RunStepValidator,
    _get_database_reference_path,
)
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.importers.dike_section_importer import DikeSectionImporter
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.orm_controllers import open_database

OptimizationStepResult = (
    orm.OptimizationStepResultMechanism | orm.OptimizationStepResultSection
)


class RunStepAssessmentValidator(RunStepValidator):
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)
        assert not any(orm.AssessmentMechanismResult.select())
        assert not any(orm.AssessmentSectionResult.select())
        if not _connected_db.is_closed():
            _connected_db.close()

    def validate_results(self, valid_vrtool_config: VrtoolConfig):
        _reference_database_path = _get_database_reference_path(valid_vrtool_config)

        def load_assessment_reliabilities(
            vrtool_db: Path,
        ) -> dict[SectionData, SectionReliability]:
            _connected_db = open_database(vrtool_db)
            _assessment_reliabilities = dict(
                (_sd, DikeSectionImporter.import_assessment_reliability(_sd))
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
        for _ref_section, _ref_reliability in _reference_assessment.items():
            _res_reliability = _result_assessment.get(_ref_section, {})
            if not _res_reliability:
                _errors.append(f"Section {_ref_section} missing in result assessments.")
                continue
            if _ref_reliability != _res_reliability:
                _errors.append(f"Section {_ref_section} reliability results differ.")
        if _errors:
            pytest.fail("\n".join(_errors))
