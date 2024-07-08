import shutil
from typing import Iterator

import pytest

from tests import get_clean_test_results_dir, test_data
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.orm.io.importers.optimization.optimization_traject_importer import (
    OptimizationTrajectImporter,
)
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models import DikeTrajectInfo as OrmDikeTrajectInfo
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result.measure_result import (
    MeasureResult as OrmMeasureResult,
)
from vrtool.orm.orm_controllers import open_database


class TestOptimizationTrajectImporter:
    def test_initialize(self):
        # 1. Run test.
        _importer = OptimizationTrajectImporter(None, None)

        # 2. Verify expectations
        assert isinstance(_importer, OptimizationTrajectImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    @pytest.fixture(name="vrtool_561_config")
    def _get_vrtool_561_traject_importer_fixture(
        self, request: pytest.FixtureRequest
    ) -> Iterator[VrtoolConfig]:
        """
        [VRTOOL-561] Sections without selected optimization measures.
        """
        # 1. Define test data.
        _test_db = test_data.joinpath(
            "reported_bugs",
            "test_sections_without_selected_measures",
            "database_16-1_testcase_no_custom.db",
        )
        assert _test_db.exists()

        # 2. Create a copy to avoid overwriting or locking data.
        _results_dir = get_clean_test_results_dir(request)
        _test_db_copy = _results_dir.joinpath("traject_importer.db")
        shutil.copy(_test_db, _test_db_copy)

        # 2. Yield configuration
        yield VrtoolConfig(
            input_database_name=_test_db_copy.name,
            input_directory=_test_db_copy.parent,
            traject="16-1",
        )

        # 3. Remove output dir (not required to analyze as we are only importing here).
        shutil.rmtree(_results_dir)

    def test_given_traject_with_multiple_sections_imports_all_section_as_input(
        self, vrtool_561_config: VrtoolConfig
    ):
        # 1. Define test data.
        assert isinstance(vrtool_561_config, VrtoolConfig)

        # We just need one case to verify the correct functioning
        _measure_results_to_import = [[1, 0]]

        # 2. Run test.
        with open_database(vrtool_561_config.input_database_path).connection_context():
            _traject_to_import = OrmDikeTrajectInfo.get(
                OrmDikeTrajectInfo.traject_name == vrtool_561_config.traject
            )
            _imported_section_as_input = OptimizationTrajectImporter(
                vrtool_561_config, _measure_results_to_import
            ).import_orm(_traject_to_import)

        # 3. Verify expectations.
        assert isinstance(_imported_section_as_input, list)
        assert len(_imported_section_as_input) == 4
        assert all(
            isinstance(_isai, SectionAsInput) for _isai in _imported_section_as_input
        )
        # Only measures for the second section were imported
        assert not any(_imported_section_as_input[0].measures)
        assert any(_imported_section_as_input[1].measures)
        assert not any(_imported_section_as_input[2].measures)
        assert not any(_imported_section_as_input[3].measures)

    def test_given_measure_with_section_not_in_analysis_crashes(
        self, vrtool_561_config: VrtoolConfig
    ):
        # 1. Define test data.
        assert isinstance(vrtool_561_config, VrtoolConfig)

        # 2. Run test.
        with open_database(vrtool_561_config.input_database_path).connection_context():
            _traject_to_import = OrmDikeTrajectInfo.get(
                OrmDikeTrajectInfo.traject_name == vrtool_561_config.traject
            )
            _ds_not_in_analysis = next(
                _ds for _ds in _traject_to_import.dike_sections if not _ds.in_analysis
            )
            _mps = MeasurePerSection.create(
                section=_ds_not_in_analysis, measure=Measure.get()
            )
            _measure_result = OrmMeasureResult.create(measure_per_section=_mps)
            _expected_error_mssg = f"Niet mogelijk om te importeren maatregel (id={_measure_result.get_id()}) in sectie (naam={_ds_not_in_analysis.section_name}) vanwege 'sectie.in_analysis' is False."

            with pytest.raises(ValueError) as exc_err:
                OptimizationTrajectImporter(
                    vrtool_561_config, [(_measure_result.get_id(), 0)]
                ).import_orm(_traject_to_import)

        # 3. Verify expectations
        assert str(exc_err.value) == _expected_error_mssg
