import pytest
import pandas as pd

from peewee import SqliteDatabase

from tests.orm.io.importers import db_fixture
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.overflow_hydra_ring_importer import OverFlowHydraRingImporter
from vrtool.orm.models.computation_scenario import ComputationScenario


class TestOverflowHydraRingImporter:
    def test_initialize(self):
        _importer = OverFlowHydraRingImporter()
        assert isinstance(_importer, OverFlowHydraRingImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(self, db_fixture: SqliteDatabase):
        # 1. Define test data.
        _importer = OverFlowHydraRingImporter()

        # 2. Run test
        _mechanism_input = _importer.import_orm(ComputationScenario.get_by_id(1))

        # 3. Verify expectations.
        assert len(_mechanism_input.input) == 3
        assert _mechanism_input.input["h_crest"] == pytest.approx(9.13)
        assert _mechanism_input.input["d_crest"] == pytest.approx(0.005)

        _mechanism_table_data = _mechanism_input.input["hc_beta"]
        assert isinstance(_mechanism_table_data, pd.DataFrame)

        assert list(_mechanism_table_data.columns) == ["2023", "2100"]
        assert _mechanism_table_data.index.to_list() == [
            9.51,
            9.76,
            10.01,
            10.26,
            10.51,
            10.76,
            11.01,
            11.26,
            11.51,
            11.76,
            12.01,
            12.26,
            12.51,
        ]

        assert list(_mechanism_table_data["2023"]) == [
            3.053,
            3.3826,
            3.7217,
            4.0096,
            4.2599,
            4.4969,
            4.7308,
            4.97,
            5.2193,
            5.452,
            5.6564,
            5.8677,
            6.0908,
        ]

        assert list(_mechanism_table_data["2100"]) == [
            2.2631,
            2.5689,
            2.9258,
            3.2309,
            3.5389,
            3.8489,
            4.1783,
            4.5517,
            5.0286,
            5.39,
            5.6481,
            6.0668,
            6.618,
        ]

    def test_import_orm_without_model_raises_value(self):
        # Setup
        _importer = OverFlowHydraRingImporter()
        _expected_mssg = "No valid value given for ComputationScenario."

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # Assert
        assert str(value_error.value) == _expected_mssg
