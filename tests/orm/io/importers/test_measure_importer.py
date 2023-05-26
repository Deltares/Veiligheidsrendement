from typing import Type
import pytest
from vrtool.decision_making.measures.measure_base import MeasureBase
from vrtool.decision_making.measures.soil_reinforcement_measure import SoilReinforcementMeasure
from vrtool.defaults.vrtool_config import VrtoolConfig
from tests import test_data, test_results
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.measure_importer import MeasureImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from tests.orm import empty_db_fixture
from peewee import SqliteDatabase
from vrtool.orm.models.combinable_type import CombinableType
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_type import MeasureType
from vrtool.orm.models.standard_measure import StandardMeasure
from collections.abc import Iterator

class TestSolutionsImporter:

    @pytest.fixture
    def valid_config(self) -> VrtoolConfig:
        _vr_config = VrtoolConfig()
        _vr_config.input_directory = test_data
        _vr_config.output_directory = test_results

        yield _vr_config

    def test_initialize(self, valid_config: VrtoolConfig):
        _importer = MeasureImporter(valid_config, DikeSection())
        assert isinstance(_importer, MeasureImporter)
        assert isinstance(_importer, OrmImporterProtocol)
    
    def _set_standard_measure(self, measure: Measure) -> StandardMeasure:
        _measure = StandardMeasure(
            measure=measure,
            crest_step = 4.2,
            direction = "onwards",
            stability_screen = 0,
            max_crest_increase = 0.1,
            max_outward_reinforcement = 0.2,
            max_inward_reinforcement = 0.3,
            prob_of_solution_failure = 0.4,
            failure_probability_with_solution = 0.5)
        _measure.save()

    def _get_valid_measure(self, measure_type: str, combinable_type: str) -> Measure:
        _measure_type = MeasureType(name=measure_type)
        _measure_type.save()
        _combinable_type = CombinableType(name=combinable_type)
        _combinable_type.save()
        _measure = Measure(measure_type = _measure_type, combinable_type=_combinable_type, name="Test Measure", year=2023)
        _measure.save()
        self._set_standard_measure(_measure)
        return _measure

    @pytest.mark.parametrize("measure_type, combinable_type, expected_type",
        [
            pytest.param("Soil reinforcement", "combinable", SoilReinforcementMeasure,id="Soil reinforcement measure.")
        ])
    def test_import_orm_with_standard_measure(self, measure_type: str, combinable_type: str, expected_type: Type[MeasureBase], valid_config: VrtoolConfig, empty_db_fixture: SqliteDatabase):
        # 1. Define test data.
        _importer = MeasureImporter(valid_config, DikeSection())
        _orm_measure = self._get_valid_measure(measure_type, combinable_type)

        # 2. Run test.
        _imported_measure = _importer.import_orm(_orm_measure)

        # 3. Verify final expectations.
        assert isinstance(_imported_measure, MeasureBase)
        assert isinstance(_imported_measure, expected_type)
    
    def test_import_orm_with_unknown_standard_measure_raises_error(self, valid_config: VrtoolConfig, empty_db_fixture: SqliteDatabase):
        # 1. Define test data.
        _importer = MeasureImporter(valid_config, DikeSection())
        _measure_type_name = "Not a valid measure"
        _orm_measure = self._get_valid_measure(_measure_type_name, "combinable")

        # 2. Run test.
        with pytest.raises(NotImplementedError) as exc_err:
            _importer.import_orm(_orm_measure)

        # 3. Verify expectations.
        assert str(exc_err.value) == f"No import available for {_measure_type_name}."

