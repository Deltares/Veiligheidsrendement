import pytest

from tests.orm import empty_db_fixture, get_basic_section_data
from tests.orm.io.exporters import (
    create_required_mechanism_per_section,
    section_reliability_with_values,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.exporters.mechanism_reliability_collection_exporter import (
    MechanismReliabilityCollectionExporter,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
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

    def test_export_dom_with_valid_arguments(
        self, section_reliability_with_values: SectionReliability, empty_db_fixture
    ):
        # 1. Define test data.
        _test_section_data = get_basic_section_data()
        assert not any(AssessmentMechanismResult.select())

        _expected_mechanisms_reliability = (
            section_reliability_with_values.SectionReliability.loc[
                section_reliability_with_values.SectionReliability.index != "Section"
            ]
        )
        _expected_time_entries = len(_expected_mechanisms_reliability.columns)
        _expected_mechanisms = _expected_mechanisms_reliability.index
        create_required_mechanism_per_section(_test_section_data, _expected_mechanisms)
        assert any(Mechanism.select())
        assert any(MechanismPerSection.select())

        # 2. Run test.
        _exporter = MechanismReliabilityCollectionExporter(_test_section_data)
        _orm_assessments = _exporter.export_dom(section_reliability_with_values)

        # 3. Verify expectations.
        assert len(_orm_assessments) == _expected_time_entries * len(
            _expected_mechanisms
        )
        assert all(
            isinstance(_orm_assessment, AssessmentMechanismResult)
            for _orm_assessment in _orm_assessments
        )
        for row_idx, mechanism_row in _expected_mechanisms_reliability.iterrows():
            _mechanism_name = row_idx.upper().strip()
            _orm_mechanisms = list(
                filter(
                    lambda x: x.mechanism_per_section.mechanism.name == _mechanism_name,
                    _orm_assessments,
                )
            )
            for time_idx, beta_value in enumerate(mechanism_row):
                time_value = int(mechanism_row.index[time_idx])
                _orm_assessment = next(
                    (_oa for _oa in _orm_mechanisms if _oa.time == time_value), None
                )
                assert isinstance(
                    _orm_assessment, AssessmentMechanismResult
                ), f"No assessment created for mechanism {_mechanism_name}, time {time_value}."
                assert _orm_assessment.beta == beta_value

    def test_export_dom_with_two_sections_exports_to_expected(
        self, section_reliability_with_values: SectionReliability, empty_db_fixture
    ):
        # 1. Define test data.
        _test_section_data = get_basic_section_data()
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
        _expected_mechanisms = _expected_mechanisms_reliability.index
        create_required_mechanism_per_section(_test_section_data, _expected_mechanisms)
        create_required_mechanism_per_section(
            _additional_section_data, _expected_mechanisms
        )
        assert any(Mechanism.select())
        assert any(MechanismPerSection.select())

        # 2. Run test.
        _exporter = MechanismReliabilityCollectionExporter(_additional_section_data)
        _orm_assessments = _exporter.export_dom(section_reliability_with_values)

        # 3. Verify expectations.
        assert len(_orm_assessments) == _expected_time_entries * len(
            _expected_mechanisms
        )
        assert all(
            isinstance(_orm_assessment, AssessmentMechanismResult)
            for _orm_assessment in _orm_assessments
        )
        assert all(
            _amr.mechanism_per_section.section == _additional_section_data
            for _amr in AssessmentMechanismResult.select()
        )

    def test_export_dom_with_unknown_mechanism_raises_error(
        self, section_reliability_with_values: SectionReliability, empty_db_fixture
    ):
        # 1. Define test data.
        _test_section_data = get_basic_section_data()
        assert not any(AssessmentMechanismResult.select())
        assert not any(Mechanism.select())

        _expected_mechanism_not_found = (
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
