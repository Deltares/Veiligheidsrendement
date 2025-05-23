from typing import Callable

import pytest

from tests.orm import with_empty_db_context
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.io.exporters.safety_assessment.mechanism_reliability_collection_exporter import (
    MechanismReliabilityCollectionExporter,
)
from vrtool.orm.models.assessment_mechanism_result import AssessmentMechanismResult
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData


class TestMechanismReliabilityCollectionExporter:
    def test_initialize(self):
        # Initialize exporter.
        _exporter = MechanismReliabilityCollectionExporter(None)

        # Verify expectations.
        assert isinstance(_exporter, MechanismReliabilityCollectionExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    @with_empty_db_context
    def test_export_dom_with_valid_arguments(
        self,
        section_reliability_with_values: SectionReliability,
        get_orm_basic_dike_section: Callable[[], SectionData],
        create_required_mechanism_per_section: Callable[
            [SectionData, list[MechanismEnum]], None
        ],
    ):
        # 1. Define test data.
        _test_section_data = get_orm_basic_dike_section()
        assert not any(AssessmentMechanismResult.select())

        _expected_mechanisms_reliability = (
            section_reliability_with_values.SectionReliability.loc[
                section_reliability_with_values.SectionReliability.index != "Section"
            ]
        )
        _expected_time_entries = len(_expected_mechanisms_reliability.columns)
        _expected_mechanisms = list(
            map(MechanismEnum.get_enum, _expected_mechanisms_reliability.index)
        )
        create_required_mechanism_per_section(_test_section_data, _expected_mechanisms)
        assert any(Mechanism.select())
        assert any(MechanismPerSection.select())

        # 2. Run test.
        MechanismReliabilityCollectionExporter(_test_section_data).export_dom(
            section_reliability_with_values
        )

        # 3. Verify expectations.
        assert len(AssessmentMechanismResult.select()) == _expected_time_entries * len(
            _expected_mechanisms
        )

        for row_idx, mechanism_row in _expected_mechanisms_reliability.iterrows():
            _mechanism = MechanismEnum.get_enum(row_idx)
            for time_idx, beta_value in enumerate(mechanism_row):
                time_value = int(mechanism_row.index[time_idx])
                _orm_assessment = (
                    AssessmentMechanismResult.select()
                    .join(MechanismPerSection)
                    .join(Mechanism)
                    .where(
                        (Mechanism.name == _mechanism.name)
                        & (MechanismPerSection.section == _test_section_data)
                        & (AssessmentMechanismResult.time == time_value)
                    )
                    .get()
                )
                assert isinstance(
                    _orm_assessment, AssessmentMechanismResult
                ), f"No assessment created for mechanism {_mechanism}, time {time_value}."
                assert _orm_assessment.beta == beta_value

    @with_empty_db_context
    def test_export_dom_with_two_sections_exports_to_expected(
        self,
        section_reliability_with_values: SectionReliability,
        get_orm_basic_dike_section: Callable[[], SectionData],
        create_required_mechanism_per_section: Callable[
            [SectionData, list[MechanismEnum]], None
        ],
    ):
        # 1. Define test data.
        _test_section_data = get_orm_basic_dike_section()
        _additional_section_data = SectionData.create(
            dike_traject=_test_section_data.dike_traject,
            section_name="AdditionalSection",
            meas_start=2.4,
            meas_end=4.2,
            section_length=123,
            in_analysis=True,
            crest_height=24,
            annual_crest_decline=42,
        )
        assert not any(AssessmentMechanismResult.select())

        _expected_mechanisms_reliability = (
            section_reliability_with_values.SectionReliability.loc[
                section_reliability_with_values.SectionReliability.index != "Section"
            ]
        )
        _expected_time_entries = len(_expected_mechanisms_reliability.columns)
        _expected_mechanisms = list(
            map(MechanismEnum.get_enum, _expected_mechanisms_reliability.index)
        )
        create_required_mechanism_per_section(_test_section_data, _expected_mechanisms)
        create_required_mechanism_per_section(
            _additional_section_data, _expected_mechanisms
        )
        assert any(Mechanism.select())
        assert any(MechanismPerSection.select())

        # 2. Run test.
        MechanismReliabilityCollectionExporter(_additional_section_data).export_dom(
            section_reliability_with_values
        )

        # 3. Verify expectations.
        assert len(AssessmentMechanismResult.select()) == _expected_time_entries * len(
            _expected_mechanisms
        )
        assert all(
            _amr.mechanism_per_section.section == _additional_section_data
            for _amr in AssessmentMechanismResult.select()
        )

    @with_empty_db_context
    def test_export_dom_with_unknown_mechanism_raises_error(
        self,
        section_reliability_with_values: SectionReliability,
        get_orm_basic_dike_section: Callable[[], SectionData],
    ):
        # 1. Define test data.
        _test_section_data = get_orm_basic_dike_section()
        assert not any(AssessmentMechanismResult.select())
        assert not any(Mechanism.select())

        _expected_mechanism_not_found = MechanismEnum.get_enum(
            section_reliability_with_values.SectionReliability.index[0]
        )

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            _exporter = MechanismReliabilityCollectionExporter(_test_section_data)
            _exporter.export_dom(section_reliability_with_values)

        # 3. Verify final expectations.
        assert (
            str(exc_err.value)
            == f"No mechanism found for {_expected_mechanism_not_found}."
        )
