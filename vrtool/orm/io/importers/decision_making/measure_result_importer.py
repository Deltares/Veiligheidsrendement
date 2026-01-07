from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.models.measure_result.measure_result_mechanism import (
    MeasureResultMechanism,
)
from vrtool.orm.models.measure_result.measure_result_section import MeasureResultSection
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class MeasureResultImporter(OrmImporterProtocol):
    @staticmethod
    def import_measure_reliability(
        measure_result: MeasureResult,
    ) -> SectionReliability:
        """
        Imports all the reliability values of a given `MeasureResult` into a
        SectionReliability object.

        Args:
            measure_result (MeasureResult): The measure result whose reliability
            needs to be imported.

        Returns:
            SectionReliability: Object containing reliability relative to the
            measure - section and measure - mechanisms.
        """
        _section_reliability = SectionReliability()
        for _smr in measure_result.measure_result_section.order_by(
            MeasureResultSection.time.asc()
        ):
            _section_reliability.set_reliability_for_year(
                _smr.time, beta_to_pf(_smr.beta)
            )
            for _mrm in measure_result.measure_result_mechanisms.where(
                MeasureResultMechanism.time == _smr.time
            ):
                _section_reliability.set_reliability_for_mechanism_year(
                    MechanismEnum.get_enum(_mrm.mechanism_per_section.mechanism.name),
                    _mrm.time,
                    beta_to_pf(_mrm.beta),
                )
        return _section_reliability

    def import_orm(self, measure_result: OrmBaseModel) -> dict:
        _cost = float("nan")
        if any(measure_result.measure_result_section):
            # The measure cost has the same value regardless of the time.
            _cost = measure_result.measure_result_section[0].cost

        _section_reliability = self.import_measure_reliability(measure_result)

        # Get measure parameters (dberm, dcrest, target_beta, transition_level, ...).
        _imported_parameters = dict(
            (mrp.name.lower(), mrp.value)
            for mrp in measure_result.measure_result_parameters
        )

        # Set attributes.
        return dict(
            measure_result_id=measure_result.get_id(),
            measure_id=measure_result.measure_per_section.measure.get_id(),
            Cost=_cost,
            Reliability=_section_reliability,
            imported_parameters=_imported_parameters,
            combinable=measure_result.measure_per_section.measure.combinable_type.name,
            reinforcement_type=measure_result.measure_per_section.measure.measure_type.name,
        )
