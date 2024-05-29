import math

import pytest

from tests import (
    get_copy_of_reference_directory,
    get_vrtool_config_test_copy,
    test_data,
)
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure
from vrtool.orm.io.importers.optimization.optimization_measure_result_importer import (
    OptimizationMeasureResultImporter,
)
from vrtool.orm.models.measure_result.measure_result import (
    MeasureResult as OrmMeasureResult,
)
from vrtool.orm.orm_controllers import open_database


class TestOptimizationMeasureResultImporter:
    def test_given_valid_case_import_all_measure_results(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _test_dir_name = "test_stability_multiple_scenarios"
        _test_case_dir = get_copy_of_reference_directory(_test_dir_name)

        _investment_years = [0]

        _vrtool_config = get_vrtool_config_test_copy(
            _test_case_dir.joinpath("config.json"), request.node.name
        )
        assert not any(_vrtool_config.output_directory.glob("*"))

        # 2. Run test.
        _importer = OptimizationMeasureResultImporter(_vrtool_config, _investment_years)

        with open_database(_vrtool_config.input_database_path).connection_context():
            _imported_results = _importer.import_orm(OrmMeasureResult.select().get())

        # 3. Verify final expectations.
        assert any(_imported_results)
        assert all(isinstance(_ir, MeasureAsInputProtocol) for _ir in _imported_results)

    @pytest.mark.fixture_database(
        test_data.joinpath("38-1 custom measures", "with_aggregated_measures.db")
    )
    def test_get_measure_as_input_types_custom_measure_returns_all_types(
        self, custom_measures_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        _expected_types = [ShMeasure, SgMeasure, ShSgMeasure]

        # 2. Run test.
        with open_database(
            custom_measures_vrtool_config.input_database_path
        ).connection_context():
            _input_types = OptimizationMeasureResultImporter.get_measure_as_input_types(
                # The provided database only contains 'Custom' MeasureResult rows.
                # so we do not need to worry further.
                OrmMeasureResult.select().get()
            )

        # 3. Verify expectations.
        assert all(_et in _input_types for _et in _expected_types)

    @pytest.mark.fixture_database(
        test_data.joinpath("38-1 custom measures", "with_aggregated_measures.db")
    )
    def test_import_orm_for_custom_measures(
        self, custom_measures_vrtool_config: VrtoolConfig
    ):
        with open_database(
            custom_measures_vrtool_config.input_database_path
        ).connection_context():
            # 1. Define test data.
            _expected_types = [ShMeasure, SgMeasure, ShSgMeasure]
            _investment_years = [0]

            # The provided database only contains 'Custom' MeasureResult rows.
            # so we do not need to worry further.
            _measure_result = OrmMeasureResult.select().get()
            _measure_result_id = _measure_result.get_id()

            # 2. Run test.

            _imported_measure_as_input_list = OptimizationMeasureResultImporter(
                custom_measures_vrtool_config, _investment_years
            ).import_orm(_measure_result)

        # 3. Verify expectations.
        assert len(_imported_measure_as_input_list) == 3

        for _imported_mai in _imported_measure_as_input_list:
            assert isinstance(_imported_mai, MeasureAsInputProtocol)
            assert _imported_mai.measure_result_id == _measure_result_id
            assert _imported_mai.measure_type == MeasureTypeEnum.CUSTOM
            assert math.isnan(_imported_mai.l_stab_screen)

        _measure_input_type = list(map(type, _imported_measure_as_input_list))
        assert all(_et in _measure_input_type for _et in _expected_types)
