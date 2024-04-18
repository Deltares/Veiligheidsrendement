import logging

from vrtool.common.enums.mechanism_enum import MechanismEnum
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

    def _get_piping_reliability_collection(
        self, dike_section: DikeSection, variable: str
    ) -> list[float]:
        piping_reliability_collection = dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
            MechanismEnum.PIPING
        )
        if not piping_reliability_collection:
            error_message = f'No Piping present for selfretaining sheetpile measure section "{dike_section.name}".'
            logging.error(error_message)
            return []

        return list(
            piping_reliability_collection.Reliability["0"].Input.input.get(variable, [])
        )

    def _get_dcover(self, dike_section: DikeSection) -> float:
        """
        VRTOOL-344 Because potentially there are multiple values for `d_cover`,
        one per scenario, we will just take the most extreme one (greatest value).

        Returns:
            float: max `d_cover` value.
        """
        _d_cover_input = self._get_piping_reliability_collection(
            dike_section, "d_cover"
        )
        # Assume we got a list
        if not _d_cover_input:
            return 1.0
        return max(_d_cover_input)

    def _get_maaiveld(self, dike_section: DikeSection) -> float:
        """
        VRTOOL-344 Because potentially there are multiple values for `h_exit`,
        one per scenario, we will just take the most extreme one (lowest value).

        Returns:
            float: min `h_exit` value.
        """
        _h_exit_input = self._get_piping_reliability_collection(dike_section, "h_exit")

        # Assume we get a list
        if not _h_exit_input:
            return dike_section.crest_height - 3
        return min(_h_exit_input)

    def _calculate_measure_costs(self, dike_section: DikeSection) -> float:
        """
        Overriden method as it is the only difference with the `DiaphragmWallMeasure`.
        """
        _h_dike = dike_section.crest_height - self._get_maaiveld(dike_section)
        _d_cover = self._get_dcover(dike_section)

        _length_sheetpile = min((_h_dike + _d_cover) * 3, 20)
        _l_section = dike_section.Length
        logging.info("Calculating selfretaining sheetpile measure costs:")
        logging.info(
            "h_dike: {}; d_cover: {}; length_sheetpile: {}; l_section: {}".format(
                _h_dike, _d_cover, _length_sheetpile, _l_section
            )
        )
        return self._selfretaining_sheetpile_cost * _length_sheetpile * _l_section
