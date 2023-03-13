from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class TestResultsMeasures:
    def test_init(self):
        _results = ResultsMeasures()

        assert isinstance(_results, ResultsMeasures)
        assert isinstance(_results, VrToolRunResultProtocol)
        assert _results.solutions_dict == {}
