from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.wall_measures.diaphragm_wall_measure import (
    DiaphragmWallMeasure,
)
from vrtool.flood_defence_system.dike_section import DikeSection


class SelfretainingSheetpileMeasure(DiaphragmWallMeasure, MeasureProtocol):
    """
    VRTOOL-344. We require a small variation of the `DiaphragmWallMeasure`.
    For this case we can simply inherit from it. Keep in mind that if further
    properties or methods are to be adapted it might be better to remove the inheritance
    and define its own methods here.

    Args:
        DiaphragmWallMeasure (MeasureProtocol): Base class containing all logic except cost calculation.
        MeasureProtocol (Protocol): Protocol to implement / adhere by this class.
    """

    def _calculate_measure_costs(self, dike_section: DikeSection) -> float:
        """
        Overriden method as it is the only difference with the `DiaphragmWallMeasure`.
        """
        return super()._calculate_measure_costs(dike_section)
