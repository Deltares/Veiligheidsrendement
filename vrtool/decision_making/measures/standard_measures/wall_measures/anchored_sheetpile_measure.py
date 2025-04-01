import logging

from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.wall_measures.diaphragm_wall_measure import (
    DiaphragmWallMeasure,
)
from vrtool.flood_defence_system.dike_section import DikeSection


class AnchoredSheetpileMeasure(DiaphragmWallMeasure, MeasureProtocol):
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
        Length of sheetpile should be between 10 and 20 meters.
        """
        _h_dike = (
            dike_section.crest_height - dike_section.InitialGeometry.loc["BIT"]["z"]
        )
        _d_cover = dike_section.cover_layer_thickness

        _length_sheetpile = max(min((_h_dike + _d_cover) * 3, 20), 10)
        _l_section = dike_section.Length
        logging.debug("Calculating anchored sheetpile measure costs:")
        logging.debug(
            "h_dike: {}; d_cover: {}; length_sheetpile: {}; l_section: {}".format(
                _h_dike, _d_cover, _length_sheetpile, _l_section
            )
        )
        return self.unit_costs.anchored_sheetpile * _length_sheetpile * _l_section
