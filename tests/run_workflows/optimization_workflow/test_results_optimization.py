from vrtool.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)
from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class TestResultsOptimization:
    def test_init(self):
        _results = ResultsOptimization()

        assert isinstance(_results, ResultsOptimization)
        assert isinstance(_results, VrToolRunResultProtocol)
        assert _results.results_strategies == []
