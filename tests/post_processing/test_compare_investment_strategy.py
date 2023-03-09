import shutil
import pytest
from pathlib import Path
import pandas as pd
from vrtool.post_processing.compare_investment_strategy import compare_investment_strategy
from tests import test_data, test_results
class TestCompareInvestmentStrategy:

    @pytest.mark.skip(reason="TODO: This test misses all the input data and results verification.")
    def test_compare_investment_strategy_with_valid_data(self, request: pytest.FixtureRequest):
        """
        This used to be the code contained at the previous `def main()` in the target file.
        """
        # 1. Define test data.
        _traject = "16-4"
        _investment_limit = 20000000

        _input_test_dir = test_data / "SAFEInput" / _traject / "Input"
        _measure_test_file = _input_test_dir / "measures.csv"
        assert _measure_test_file.exists(), "Measure test file not found at {}".format(_measure_test_file)        

        _tc_test_file = _input_test_dir / "TakenMeasures_TC.csv"
        assert _tc_test_file.exists(), "Not found TC measures file {}".format(_tc_test_file)

        _oi_test_file = _input_test_dir / "TakenMeasures_OI.csv"
        assert _oi_test_file.exists(), "Not found OI measures file {}".format(_oi_test_file)

        # Cleanse / generate the results dir.        
        _results_dir = test_results / request.node.name / "results_{}".format(_traject)
        if _results_dir.exists():
            shutil.rmtree(_results_dir)
        _results_dir.mkdir(parents=True)

        _xlsx_results_filename = (
            "TC_OI_comparison_traject_{}_investment_limit_{}.xlsx".format(_traject, _investment_limit)
        )
        _xlsx_results_file = _results_dir / _xlsx_results_filename
        if _xlsx_results_file.exists():
            _xlsx_results_file.unlink()

        # 2. Run test.
        # Compare TC and OI investment limit
        compare_investment_strategy(
            pd.read_csv(_results_dir / _tc_test_file, index_col=[0], skiprows=[1]),
            pd.read_csv(_results_dir / _oi_test_file, index_col=[0], skiprows=[1]),
            _investment_limit,
            _xlsx_results_file,
            _measure_test_file,
        )

        # 3. Verify expectations.
        assert _xlsx_results_file.exists(), "No results file was generated."