from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.orm.io.importers.dstability_importer import DStabilityImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.overflow_hydra_ring_importer import (
    OverFlowHydraRingImporter,
)
from vrtool.orm.io.importers.piping_importer import PipingImporter
from vrtool.orm.io.importers.stability_inner_simple_importer import (
    StabilityInnerSimpleImporter,
)
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.mechanism_per_section import MechanismPerSection


class MechanismReliabilityCollectionImporter(OrmImporterProtocol):
    def __init__(
        self,
        vrtool_config: VrtoolConfig,
    ) -> None:
        """Creates a new instance of the MechanismReliabilityCollectionImporter

        Args:
            mechanism (str): The name of the mechanism.
            computation_type (str): The computation type.
            computation_years (list[int]): The collection of years to compute the reliability for.
            t_0 (float): The initial year.
            measure_year (int): The year to compute the measure for
        """
        self.computation_years = vrtool_config.T
        self.t_0 = vrtool_config.t_0
        self.externals_path = vrtool_config.externals
        self.stix_files_directory = vrtool_config.input_directory.joinpath("stix")

        self._overflow_hydra_ring_importer = OverFlowHydraRingImporter()
        self._piping_importer = PipingImporter()
        self._stability_inner_simple_importer = StabilityInnerSimpleImporter()

        # TODO Adjust DStability importer to set externals and stix directory
        self._dstability_importer = DStabilityImporter()

    def import_orm(self, orm: MechanismPerSection) -> MechanismReliabilityCollection:
        mechanism_name = orm.mechanism.name

        # Assume computation type is the same accross the computation scenarios
        computation_scenarios = orm.computation_scenarios.select()
        computation_type = computation_scenarios[0].computation_type.name
        collection = MechanismReliabilityCollection(
            mechanism_name, computation_type, self.computation_years, self.t_0, 0
        )

        mechanism_input = self._get_mechanism_input(
            computation_scenarios, mechanism_name, computation_type
        )

        for year in collection.Reliability.keys():
            collection.Reliability[year].Input = mechanism_input

        return collection

    def _get_mechanism_input(
        self,
        computation_scenarios: list[ComputationScenario],
        mechanism: str,
        computation_type: str,
    ) -> MechanismInput:
        if mechanism == "Overflow":
            return self._overflow_hydra_ring_importer.import_orm(
                computation_scenarios[0]
            )

        if mechanism == "StabilityInner" and computation_type == "Simple":
            return self._stability_inner_simple_importer.import_orm(
                computation_scenarios[0]
            )

        if mechanism == "StabilityInner" and computation_type == "DStability":
            return self._dstability_importer.import_orm(computation_scenarios[0])

        if mechanism == "Piping":
            return self._piping_importer.import_orm(computation_scenarios)

        raise ValueError(f"Mechanism {mechanism} not supported.")
