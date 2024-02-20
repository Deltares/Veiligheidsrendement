from tests import test_data
from vrtool.orm.io.importers.measures.measure_result_importer import (
    MeasureResultImporter,
)
from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.orm_controllers import open_database


class TestMeasureResultImporter:
    def test_given_valid_measure_result_import_as_expected(self):
        # 1. Define test data.
        _test_db = test_data.joinpath("38-1 base river case", "vrtool_input.db")
        assert _test_db.is_file()

        _db = open_database(_test_db)
        _measure_result = MeasureResult.get_or_none()
        assert isinstance(
            _measure_result, MeasureResult
        ), "No MeasureResult was found in the db."

        if not _db.is_closed():
            _db.close()

        # 2. Run test.
        _result = MeasureResultImporter().import_orm(_measure_result)

        # 3. Verify expectations.
        assert isinstance(_result, dict)
