import pandas as pd
import pytest
from peewee import SqliteDatabase

from tests import test_data, test_results
from tests.orm import empty_db_fixture
from tests.orm.io.importers.measures.test_measure_importer import (
    _get_valid_measure,
    _set_custom_measure,
    _set_standard_measure,
)
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.measures.solutions_importer import SolutionsImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.section_data import SectionData


class TestSolutionsImporter:
    @pytest.fixture
    def valid_config(self) -> VrtoolConfig:
        _vr_config = VrtoolConfig()
        _vr_config.input_directory = test_data
        _vr_config.output_directory = test_results

        return _vr_config

    def test_initialize(self, valid_config: VrtoolConfig):
        _importer = SolutionsImporter(valid_config, DikeSection())
        assert isinstance(_importer, SolutionsImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_initialize_given_no_vrtoolconfig_raises_valueerror(self):
        with pytest.raises(ValueError) as exc_err:
            SolutionsImporter(None, DikeSection())

        assert str(exc_err.value) == "VrtoolConfig not provided."

    def test_initialize_given_no_dike_section_raises_valueerror(
        self, valid_config: VrtoolConfig
    ):
        with pytest.raises(ValueError) as exc_err:
            SolutionsImporter(valid_config, None)

        assert str(exc_err.value) == "DikeSection not provided."

    def test_import_orm_given_no_orm_model_raises_valueerror(
        self, valid_config: VrtoolConfig
    ):
        # 1. Define test data.
        _importer = SolutionsImporter(valid_config, DikeSection())

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            _importer.import_orm(None)

        # 3. Verify expectations.
        assert str(exc_err.value) == f"No valid value given for SectionData."

    @pytest.fixture
    def valid_section_data_without_measures(self, empty_db_fixture: SqliteDatabase):
        _traject = DikeTrajectInfo.create(traject_name="A traject")
        _section_data = SectionData.create(
            dike_traject=_traject,
            section_name="E4E5",
            meas_start=1.2,
            meas_end=2.3,
            section_length=3.4,
            in_analysis=True,
            crest_height=4.5,
            annual_crest_decline=5.6,
        )
        return _section_data

    def test_given_different_sectiondata_and_dikesection_raises_valueerror(
        self,
        valid_config: VrtoolConfig,
        valid_section_data_without_measures: SectionData,
    ):
        # 1. Define test data.
        _dike_section = DikeSection()
        _dike_section.name = "ACDC"
        _importer = SolutionsImporter(valid_config, _dike_section)
        _expected_error = "The provided SectionData ({}) does not match the given DikeSection ({}).".format(
            valid_section_data_without_measures.section_name, _dike_section.name
        )

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            _importer.import_orm(valid_section_data_without_measures)

        # 3. Verify expectations.
        assert str(exc_err.value) == _expected_error

    def test_given_section_without_measures_doesnot_raise(
        self,
        valid_config: VrtoolConfig,
        valid_section_data_without_measures: SectionData,
    ):
        # 1. Define test data.
        _dike_section = DikeSection()
        _dike_section.name = valid_section_data_without_measures.section_name
        _dike_section.Length = 42
        _dike_section.InitialGeometry = pd.DataFrame.from_dict(
            {"x": [0, 1, 2, 3, 4], "y": [4, 3, 2, 1, 0]}
        )
        _importer = SolutionsImporter(valid_config, _dike_section)

        # 2. Run test.
        _imported_solution = _importer.import_orm(valid_section_data_without_measures)

        # 3. Verify expectations.
        assert isinstance(_imported_solution, Solutions)
        assert _imported_solution.section_name == _dike_section.name
        assert _imported_solution.length == _dike_section.Length
        assert _imported_solution.initial_geometry.equals(_dike_section.InitialGeometry)
        assert _imported_solution.config == valid_config
        assert _imported_solution.T == valid_config.T
        assert _imported_solution.mechanisms == valid_config.mechanisms
        assert not any(_imported_solution.measures)

    @pytest.fixture
    def valid_section_data_with_measures(
        self, valid_section_data_without_measures: SectionData
    ) -> SectionData:
        _standard_measure = _get_valid_measure(
            "Soil reinforcement", "combinable", _set_standard_measure
        )
        _custom_measure = _get_valid_measure("Custom", "full", _set_custom_measure)

        MeasurePerSection.create(
            section=valid_section_data_without_measures, measure=_standard_measure
        )
        MeasurePerSection.create(
            section=valid_section_data_without_measures, measure=_custom_measure
        )

        return valid_section_data_without_measures

    def test_given_section_with_measures_imports_them_all(
        self, valid_config: VrtoolConfig, valid_section_data_with_measures: SectionData
    ):
        # 1. Define test data.
        _dike_section = DikeSection()
        _dike_section.name = valid_section_data_with_measures.section_name
        _dike_section.Length = 42
        _dike_section.InitialGeometry = pd.DataFrame.from_dict(
            {"x": [0, 1, 2, 3, 4], "y": [4, 3, 2, 1, 0]}
        )
        _importer = SolutionsImporter(valid_config, _dike_section)

        # 2. Run test.
        _imported_solution = _importer.import_orm(valid_section_data_with_measures)

        # 3. Verify expectations.
        assert isinstance(_imported_solution, Solutions)
        assert _imported_solution.section_name == _dike_section.name
        assert _imported_solution.length == _dike_section.Length
        assert _imported_solution.initial_geometry.equals(_dike_section.InitialGeometry)
        assert _imported_solution.config == valid_config
        assert _imported_solution.T == valid_config.T
        assert _imported_solution.mechanisms == valid_config.mechanisms
        assert len(_imported_solution.measures) == 1
        assert _imported_solution.measures[0].parameters["Type"] == "Soil reinforcement"
