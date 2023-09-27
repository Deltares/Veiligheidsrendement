from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol


class StrategyBaseExporter(OrmExporterProtocol):
    """
    TODO: Placeholder for issue VRTOOL-268.
    """
    def export_dom(self, dom_model: StrategyBase) -> None:
        raise NotImplementedError()
