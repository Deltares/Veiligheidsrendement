from pathlib import Path

from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.orm.io.importers.dstability_importer import DStabilityImporter
from vrtool.orm.io.importers.mechanism_simple_importer import MechanismSimpleImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.overflow_hydra_ring_importer import (
    OverFlowHydraRingImporter,
)
from vrtool.orm.io.importers.piping_importer import PipingImporter
from vrtool.orm.io.importers.revetment_importer import RevetmentImporter
from vrtool.orm.models.mechanism_per_section import MechanismPerSection


class MechanismReliabilityCollectionImporter(OrmImporterProtocol):
    computation_years: list[int]
    t_0: int
    externals_path: Path
    input_directory: Path

    def __init__(
        self,
        vrtool_config: VrtoolConfig,
    ) -> None:
        """Creates a new instance of the MechanismReliabilityCollectionImporter

        Args:
            vrtool_config (VrtoolConfig): A valid `VrtoolConfig` object.
        """
        self.computation_years = vrtool_config.T
        self.t_0 = vrtool_config.t_0
        self.externals_path = vrtool_config.externals
        self.input_directory = vrtool_config.input_directory

    def import_orm(
        self, orm_model: MechanismPerSection
    ) -> MechanismReliabilityCollection:
        if not orm_model:
            raise ValueError(
                f"No valid value given for {MechanismPerSection.__name__}."
            )

        mechanism = MechanismEnum.get_enum(orm_model.mechanism.name)

        # Assume computation type is the same accross the computation scenarios
        computation_scenarios = orm_model.computation_scenarios.select()
        computation_type = ComputationTypeEnum.get_enum(
            computation_scenarios[0].computation_type.name
        )
        collection = MechanismReliabilityCollection(
            mechanism, computation_type, self.computation_years, self.t_0, 0
        )

        mechanism_input = self._get_mechanism_input(
            orm_model, mechanism, computation_type
        )

        for year in collection.Reliability.keys():
            collection.Reliability[year].Input = mechanism_input

        return collection

    def _get_mechanism_input(
        self,
        mechanism_per_section: MechanismPerSection,
        mechanism: MechanismEnum,
        computation_type: ComputationTypeEnum,
    ) -> MechanismInput:
        # TODO: VRTOOL-340 - no multiple computation scenarios possible from here on.

        if mechanism == MechanismEnum.OVERFLOW:
            # TODO: VRTOOL-340. This does not support multiple scenarios.
            return OverFlowHydraRingImporter().import_orm(
                mechanism_per_section.computation_scenarios.select().get()
            )

        if mechanism == MechanismEnum.STABILITY_INNER:
            if computation_type == ComputationTypeEnum.SIMPLE:
                return MechanismSimpleImporter().import_orm(mechanism_per_section)

            elif computation_type == ComputationTypeEnum.DSTABILITY:
                _dstability_importer = DStabilityImporter(
                    self.externals_path, self.input_directory.joinpath("stix")
                )
                return _dstability_importer.import_orm(mechanism_per_section)

        if mechanism == MechanismEnum.PIPING:
            if computation_type == ComputationTypeEnum.SIMPLE:
                return MechanismSimpleImporter().import_orm(mechanism_per_section)
            return PipingImporter().import_orm(mechanism_per_section)

        if mechanism == MechanismEnum.REVETMENT:
            # TODO: VRTOOL-340. This does not support multiple scenarios.
            return RevetmentImporter().import_orm(
                mechanism_per_section.computation_scenarios.select().get()
            )

        raise ValueError(f"Mechanism {mechanism} not supported.")
