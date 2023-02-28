from src.run_workflows.measures_workflow.run_measures import RunMeasures
from src.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class TestRunMeasures:
    def test_init(self):
        _run = RunMeasures("sth")

        assert isinstance(_run, RunMeasures)
        assert isinstance(_run, VrToolRunProtocol)
