from typing import Callable, Type

import pytest

from tests import test_data, test_results
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.decision_making.measures import (
    DiaphragmWallMeasure,
    SoilReinforcementMeasure,
    StabilityScreenMeasure,
    VerticalPipingSolutionMeasure,
)
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.revetment_measure import (
    RevetmentMeasure,
)
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.io.importers.decision_making.measure_importer import MeasureImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.measure import Measure


class TestMeasureImporter:
    @pytest.fixture
    def valid_config(self) -> VrtoolConfig:
        _vr_config = VrtoolConfig()
        _vr_config.input_directory = test_data
        _vr_config.output_directory = test_results
        _vr_config.berm_step = 4.2
        _vr_config.t_0 = 42
        _vr_config.unit_costs = MeasureUnitCosts(
            **dict(
                inward_added_volume=float("nan"),
                inward_starting_costs=float("nan"),
                outward_reuse_factor=float("nan"),
                outward_removed_volume=float("nan"),
                outward_reused_volume=float("nan"),
                outward_added_volume=float("nan"),
                outward_compensation_factor=float("nan"),
                house_removal=float("nan"),
                road_renewal=float("nan"),
            )
        )
        return _vr_config

    def test_initialize(self, valid_config: VrtoolConfig):
        _importer = MeasureImporter(valid_config)
        assert isinstance(_importer, MeasureImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_initialize_given_no_vrtoolconfig_raises_valueerror(self):
        with pytest.raises(ValueError) as exc_err:
            MeasureImporter(None)
        assert str(exc_err.value) == "VrtoolConfig not provided."

    def test_import_orm_given_no_orm_model_raises_valueerror(
        self, valid_config: VrtoolConfig
    ):
        # 1. Define test data.
        _importer = MeasureImporter(valid_config)

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            _importer.import_orm(None)

        # 3. Verify expectations.
        assert str(exc_err.value) == f"No valid value given for Measure."

    @pytest.mark.parametrize(
        "measure_type, expected_type",
        [
            pytest.param(
                MeasureTypeEnum.SOIL_REINFORCEMENT,
                SoilReinforcementMeasure,
                id="Soil Reinforcement measure.",
            ),
            pytest.param(
                MeasureTypeEnum.DIAPHRAGM_WALL,
                DiaphragmWallMeasure,
                id="Diaphragm Wall measure.",
            ),
            pytest.param(
                MeasureTypeEnum.STABILITY_SCREEN,
                StabilityScreenMeasure,
                id="Stability Screen measure.",
            ),
            pytest.param(
                MeasureTypeEnum.VERTICAL_PIPING_SOLUTION,
                VerticalPipingSolutionMeasure,
                id="Vertical Piping Solution measure.",
            ),
            pytest.param(
                MeasureTypeEnum.REVETMENT, RevetmentMeasure, id="Revetment measure"
            ),
        ],
    )
    @pytest.mark.parametrize(
        "combinable_type",
        [
            pytest.param(CombinableTypeEnum.COMBINABLE),
            pytest.param(CombinableTypeEnum.PARTIAL),
            pytest.param(CombinableTypeEnum.FULL),
        ],
    )
    @pytest.mark.usefixtures("empty_db_fixture")
    def test_import_orm_with_standard_measure(
        self,
        measure_type: MeasureTypeEnum,
        combinable_type: CombinableTypeEnum,
        expected_type: Type[MeasureProtocol],
        valid_config: VrtoolConfig,
        create_valid_measure: Callable[[MeasureTypeEnum, CombinableTypeEnum], Measure],
    ):
        # 1. Define test data.
        _importer = MeasureImporter(valid_config)
        _orm_measure = create_valid_measure(measure_type, combinable_type)

        # 2. Run test.
        _imported_measure = _importer.import_orm(_orm_measure)

        # 3. Verify final expectations.
        assert isinstance(_imported_measure, expected_type)
        self._validate_measure_base_values(_imported_measure, valid_config)
        assert _imported_measure.parameters["Type"] == measure_type.legacy_name
        assert _imported_measure.parameters["Direction"] == "onwards"
        assert _imported_measure.parameters["StabilityScreen"] == "no"
        assert _imported_measure.parameters["dcrest_min"] == 0
        assert _imported_measure.parameters["dcrest_max"] == 0.1
        assert _imported_measure.parameters["max_outward"] == 2
        assert _imported_measure.parameters["max_inward"] == 3
        assert _imported_measure.parameters["year"] == 2023
        assert _imported_measure.parameters["P_solution"] == 0.4
        assert _imported_measure.parameters["Pf_solution"] == 0.5
        assert (
            _imported_measure.parameters["ID"]
            == _orm_measure.standard_measure[0].get_id()
        )

    @pytest.mark.usefixtures("valid_section_data_without_measures")
    def test_import_custom_measure_raises(
        self,
        valid_config: VrtoolConfig,
        create_valid_measure: Callable[[MeasureTypeEnum, CombinableTypeEnum], Measure],
    ):
        # 1. Define test data.
        _importer = MeasureImporter(valid_config)
        _orm_measure = create_valid_measure(
            MeasureTypeEnum.CUSTOM, CombinableTypeEnum.COMBINABLE
        )
        _expected_error = "Custom measures are not supported by this importer."

        # 2. Run test.
        with pytest.raises(NotImplementedError) as _exc_err:
            _importer.import_orm(_orm_measure)

        # 3. Verify expectations.
        assert str(_exc_err.value) == _expected_error

    def _validate_measure_base_values(
        self, measure_base: MeasureProtocol, valid_config: VrtoolConfig
    ):
        assert isinstance(measure_base, MeasureProtocol)
        assert measure_base.config == valid_config
        assert measure_base.berm_step == 4.2
        assert measure_base.t_0 == 42
        assert isinstance(measure_base.unit_costs, MeasureUnitCosts)
        assert measure_base.unit_costs == valid_config.unit_costs

    @pytest.mark.usefixtures("empty_db_fixture")
    def test_import_orm_with_unknown_standard_measure_raises_error(
        self,
        valid_config: VrtoolConfig,
        create_valid_measure: Callable[[MeasureTypeEnum, CombinableTypeEnum], Measure],
    ):
        # 1. Define test data.
        _importer = MeasureImporter(valid_config)
        _measure_type = MeasureTypeEnum.INVALID
        _orm_measure = create_valid_measure(
            _measure_type, CombinableTypeEnum.COMBINABLE
        )

        # 2. Run test.
        with pytest.raises(NotImplementedError) as exc_err:
            _importer.import_orm(_orm_measure)

        # 3. Verify expectations.
        assert (
            str(exc_err.value)
            == f"No import available for {_measure_type.legacy_name}."
        )
