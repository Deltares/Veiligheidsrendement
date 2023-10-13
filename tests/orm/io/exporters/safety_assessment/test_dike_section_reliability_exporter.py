from tests.orm import empty_db_fixture, get_basic_section_data
from tests.orm.io.exporters import (
    create_required_mechanism_per_section,
    section_reliability_with_values,
)
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums import MechanismEnum
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.io.exporters.safety_assessment.dike_section_reliability_exporter import (
    DikeSectionReliabilityExporter,
)
from vrtool.orm.models.assessment_mechanism_result import AssessmentMechanismResult
from vrtool.orm.models.assessment_section_result import AssessmentSectionResult
from vrtool.orm.models.section_data import SectionData


class TestDikeSectionReliabilityExporter:
    def test_initialize(self):
        _exporter = DikeSectionReliabilityExporter()

        # Verify expectations.
        assert isinstance(_exporter, DikeSectionReliabilityExporter)
        assert isinstance(_exporter, OrmExporterProtocol)

    def _get_valid_dike_section(
        self, section_name: str, traject_name: str
    ) -> DikeSection:
        _dike_section = DikeSection()
        _dike_section.name = section_name
        _dike_section.TrajectInfo = DikeTrajectInfo(traject_name)
        return _dike_section

    def test_get_related_section_data(self, empty_db_fixture):
        # 1. Define test data.
        _test_section_data = get_basic_section_data()
        _dike_section = self._get_valid_dike_section(
            _test_section_data.section_name,
            _test_section_data.dike_traject.traject_name,
        )

        # 2. Run test.
        _related_section_data = DikeSectionReliabilityExporter.get_related_section_data(
            _dike_section
        )

        # 3. Verify expectations.
        assert isinstance(_related_section_data, SectionData)
        assert _test_section_data == _related_section_data

    def test_get_related_section_data_returns_none_for_different_traject(
        self, empty_db_fixture
    ):
        # 1. Define test data.
        _test_section_data = get_basic_section_data()
        _dike_section = self._get_valid_dike_section(
            _test_section_data.section_name,
            _test_section_data.dike_traject.traject_name,
        )
        _dike_section.TrajectInfo = DikeTrajectInfo("not_the_traject_in_the_orm")
        assert _dike_section.name == _test_section_data.section_name

        # 2. Run test.
        _related_section_data = DikeSectionReliabilityExporter.get_related_section_data(
            _dike_section
        )

        # 3. Verify expectations.
        assert _related_section_data is None

    def test_get_related_section_data_returns_none_for_different_section(
        self, empty_db_fixture
    ):
        # 1. Define test data.
        _test_section_data = get_basic_section_data()
        _dike_section = self._get_valid_dike_section(
            "not_the_section_in_the_orm",
            _test_section_data.dike_traject.traject_name,
        )
        assert (
            _dike_section.TrajectInfo.traject_name
            == _test_section_data.dike_traject.traject_name
        )

        # 2. Run test.
        _related_section_data = DikeSectionReliabilityExporter.get_related_section_data(
            _dike_section
        )

        # 3. Verify expectations.
        assert _related_section_data is None

    def test_export_dom_with_valid_data(
        self, section_reliability_with_values: SectionReliability, empty_db_fixture
    ):
        # 1. Define test data.
        _exporter = DikeSectionReliabilityExporter()
        _test_section_data = get_basic_section_data()
        _test_dike_section = self._get_valid_dike_section(
            _test_section_data.section_name,
            _test_section_data.dike_traject.traject_name,
        )
        _test_dike_section.section_reliability = section_reliability_with_values
        _expected_mechanisms_reliability = (
            section_reliability_with_values.SectionReliability.loc[
                section_reliability_with_values.SectionReliability.index != "Section"
            ]
        )
        _mechanisms = list(
            map(MechanismEnum.get_enum, _expected_mechanisms_reliability.index)
        )
        create_required_mechanism_per_section(_test_section_data, _mechanisms)

        _time_entries = len(section_reliability_with_values.SectionReliability.columns)

        # 2. Run test.
        _exporter.export_dom(_test_dike_section)

        # 3. Verify final expectations.
        _assessment_mechanisms = len(AssessmentMechanismResult.select())
        _assessment_section_data = len(AssessmentSectionResult.select())
        assert (
            _assessment_mechanisms + _assessment_section_data
            == section_reliability_with_values.SectionReliability.size
        )
        assert (
            _assessment_mechanisms
            == len(_expected_mechanisms_reliability.index) * _time_entries
        )
        assert _assessment_section_data == _time_entries
