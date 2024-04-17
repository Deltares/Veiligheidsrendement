import logging

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.common_functions import get_stability_inner_depth
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

    _selfretaining_sheetpile_cost = 1100.0

    def _calculate_measure_costs(self, dike_section: DikeSection) -> float:
        """
        Overriden method as it is the only difference with the `DiaphragmWallMeasure`.
        """
        _maaiveld = 0.42
        _h_dike = dike_section.crest_height - _maaiveld
        _d_cover = get_stability_inner_depth(dike_section)
        _length_sheetpile = min((_h_dike + _d_cover) * 3, 20)
        _l_section = dike_section.Length
        return self._selfretaining_sheetpile_cost * _length_sheetpile * _l_section
