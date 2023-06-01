from typing import Callable, Type

import pytest
from peewee import SqliteDatabase

from tests import test_data, test_results
from tests.orm import empty_db_fixture
from vrtool.decision_making.measures import (
    DiaphragmWallMeasure,
    SoilReinforcementMeasure,
    StabilityScreenMeasure,
    VerticalGeotextileMeasure,
)
from vrtool.decision_making.measures.custom_measure import CustomMeasure
from vrtool.decision_making.measures.measure_base import MeasureBase
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.io.importers.measure_importer import MeasureImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.combinable_type import CombinableType
from vrtool.orm.models.custom_measure import CustomMeasure as OrmCustomMeasure
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_parameter import MeasureParameter
from vrtool.orm.models.measure_type import MeasureType
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.standard_measure import StandardMeasure


def _set_standard_measure(measure: Measure) -> None:
    StandardMeasure.create(
        measure=measure,
        crest_step=4.2,
        direction="onwards",
        stability_screen=False,
        max_crest_increase=0.1,
        max_outward_reinforcement=2,
        max_inward_reinforcement=3,
        prob_of_solution_failure=0.4,
        failure_probability_with_solution=0.5,
    )


def _set_custom_measure(measure: Measure) -> None:
    _mechanism = Mechanism.create(name="Just a mechanism")
    _measure = OrmCustomMeasure.create(
        measure=measure,
        mechanism=_mechanism,
        cost=1234.56,
        beta=42.24,
        year=2023,
    )
    MeasureParameter.create(
        custom_measure=_measure, parameter="DummyParameter", value=24.42
    )


def _get_valid_measure(
    measure_type: str, combinable_type: str, set_measure: Callable
) -> Measure:
    _measure_type = MeasureType.create(name=measure_type)
    _combinable_type = CombinableType.create(name=combinable_type)
    _measure = Measure.create(
        measure_type=_measure_type,
        combinable_type=_combinable_type,
        name="Test Measure",
        year=2023,
    )
    set_measure(_measure)
    return _measure


class TestMeasureImporter:
    @pytest.fixture
    def valid_config(self) -> VrtoolConfig:
        _vr_config = VrtoolConfig()
        _vr_config.input_directory = test_data
        _vr_config.output_directory = test_results
        _vr_config.berm_step = 4.2
        _vr_config.t_0 = 42
        _vr_config.geometry_plot = True
        _vr_config.unit_costs = {"lorem ipsum": 123}
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
                "Soil reinforcement",
                SoilReinforcementMeasure,
                id="Soil Reinforcement measure.",
            ),
            pytest.param(
                "Diaphragm wall", DiaphragmWallMeasure, id="Diaphragm Wall measure."
            ),
            pytest.param(
                "Stability Screen",
                StabilityScreenMeasure,
                id="Stability Screen measure.",
            ),
            pytest.param(
                "Vertical Geotextile",
                VerticalGeotextileMeasure,
                id="Vertical Geotextile measure.",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "combinable_type",
        [
            pytest.param("combinable"),
            pytest.param("partial"),
            pytest.param("full"),
        ],
    )
    def test_import_orm_with_standard_measure(
        self,
        measure_type: str,
        combinable_type: str,
        expected_type: Type[MeasureBase],
        valid_config: VrtoolConfig,
        empty_db_fixture: SqliteDatabase,
    ):
        # 1. Define test data.
        _importer = MeasureImporter(valid_config)
        _orm_measure = _get_valid_measure(
            measure_type, combinable_type, _set_standard_measure
        )

        # 2. Run test.
        _imported_measure = _importer.import_orm(_orm_measure)

        # 3. Verify final expectations.
        assert isinstance(_imported_measure, expected_type)
        self._validate_measure_base_values(_imported_measure, valid_config)
        assert _imported_measure.parameters["Type"] == measure_type
        assert _imported_measure.parameters["Direction"] == "onwards"
        assert _imported_measure.parameters["StabilityScreen"] == "no"
        assert _imported_measure.parameters["dcrest_min"] == None
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

    def test_import_custom_measure(
        self, valid_config: VrtoolConfig, empty_db_fixture: SqliteDatabase
    ):
        # 1. Define test data.
        _importer = MeasureImporter(valid_config)
        _measure_type_name = "Custom"
        _orm_measure = _get_valid_measure(
            _measure_type_name, "combinable", _set_custom_measure
        )

        # 2. Run test.
        _imported_measure = _importer.import_orm(_orm_measure)

        # 3. Verify expectations.
        assert isinstance(_imported_measure, CustomMeasure)
        self._validate_measure_base_values(_imported_measure, valid_config)
        _imported_measure.measures["Cost"] == 1234.56
        _imported_measure.measures["Reliability"] == 42.24
        _imported_measure.parameters["year"] == 2023
        _imported_measure.parameters["DummyParameter"] == 24.42

    def _validate_measure_base_values(
        self, measure_base: MeasureBase, valid_config: VrtoolConfig
    ):
        assert isinstance(measure_base, MeasureBase)
        assert measure_base.config == valid_config
        assert measure_base.berm_step == 4.2
        assert measure_base.t_0 == 42
        assert measure_base.geometry_plot
        assert measure_base.unit_costs == {"lorem ipsum": 123}

    def test_import_orm_with_unknown_standard_measure_raises_error(
        self, valid_config: VrtoolConfig, empty_db_fixture: SqliteDatabase
    ):
        # 1. Define test data.
        _importer = MeasureImporter(valid_config)
        _measure_type_name = "Not a valid measure"
        _orm_measure = _get_valid_measure(
            _measure_type_name, "combinable", _set_standard_measure
        )

        # 2. Run test.
        with pytest.raises(NotImplementedError) as exc_err:
            _importer.import_orm(_orm_measure)

        # 3. Verify expectations.
        assert str(exc_err.value) == f"No import available for {_measure_type_name}."
