import pytest

from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.optimization_workflow.run_optimization import RunOptimization
from vrtool.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class TestRunOptimization:
    def test_init_with_valid_data(self):
        # 1. Define test data
        _results_measures = ResultsMeasures()
        _results_measures.vr_config = "sth"
        _results_measures.solutions_dict = 123
        _results_measures.selected_traject = 456

        # 2. Run test.
        _run = RunOptimization(_results_measures, "sth_else")

        # 3. Verify expectations.
        assert isinstance(_run, RunOptimization)
        assert isinstance(_run, VrToolRunProtocol)
        assert _run._solutions_dict == _results_measures.solutions_dict
        assert _run.selected_traject == _results_measures.selected_traject

    def test_init_with_invalid_data(self):
        with pytest.raises(ValueError) as exc_err:
            RunOptimization("not a result instance", "doh")

        assert (
            str(exc_err.value)
            == "Required valid instance of ResultsMeasures as an argument."
        )
