import copy

import numpy as np

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class VerticalPipingMeasureCalculatorBase:

    traject_info: DikeTrajectInfo
    dike_section: DikeSection
    reliability_years: list[int]
    computation_year_start: int
    measure_year: int
    measure_p_solution: float
    measure_pf_solution: float

    @classmethod
    def from_measure_section_traject(
        cls,
        measure: MeasureProtocol,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
    ):
        _calculator = cls()
        _calculator.traject_info = traject_info
        _calculator.dike_section = dike_section
        _calculator.reliability_years = measure.config.T
        _calculator.computation_year_start = measure.config.t_0
        _calculator.measure_year = measure.parameters["year"]
        _calculator.measure_p_solution = measure.parameters["P_solution"]
        _calculator.measure_pf_solution = measure.parameters["Pf_solution"]
        return _calculator

    def _get_configured_section_reliability(self) -> SectionReliability:
        section_reliability = SectionReliability()

        mechanisms = (
            self.dike_section.section_reliability.failure_mechanisms.get_available_mechanisms()
        )
        for mechanism in mechanisms:
            calc_type = self.dike_section.mechanism_data[mechanism][0][1]
            mechanism_reliability_collection = (
                self._get_configured_mechanism_reliability_collection(
                    mechanism, calc_type
                )
            )
            section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
                mechanism_reliability_collection
            )

        section_reliability.calculate_section_reliability()
        return section_reliability

    def _get_configured_mechanism_reliability_collection(
        self,
        mechanism: MechanismEnum,
        calc_type: str,
    ) -> MechanismReliabilityCollection:
        mechanism_reliability_collection = MechanismReliabilityCollection(
            mechanism, calc_type, self.reliability_years, self.computation_year_start, 0
        )

        for (
            _year_to_calculate,
            _mechanism_reliability,
        ) in mechanism_reliability_collection.Reliability.items():
            _mechanism_reliability.Input = copy.deepcopy(
                self.dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                    mechanism
                )
                .Reliability[_year_to_calculate]
                .Input
            )

            dike_section_mechanism_reliability = self.dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                mechanism
            ).Reliability[
                _year_to_calculate
            ]
            if mechanism == MechanismEnum.PIPING:
                self._configure_piping(
                    _mechanism_reliability,
                    _year_to_calculate,
                    dike_section_mechanism_reliability,
                )
            if mechanism in [MechanismEnum.OVERFLOW, MechanismEnum.STABILITY_INNER]:
                self._copy_results(
                    _mechanism_reliability, dike_section_mechanism_reliability
                )

        mechanism_reliability_collection.generate_LCR_profile(
            self.dike_section.section_reliability.load,
            traject_info=self.traject_info,
        )

        return mechanism_reliability_collection

    def _copy_results(
        self, target: MechanismReliability, source_input: MechanismReliability
    ) -> None:
        target.Input = copy.deepcopy(source_input.Input)

    def _configure_piping(
        self,
        mechanism_reliability: MechanismReliability,
        year_to_calculate: str,
        dike_section_piping_reliability: MechanismReliability,
    ) -> None:
        if int(year_to_calculate) < self.measure_year:
            self._copy_results(mechanism_reliability, dike_section_piping_reliability)

        mechanism_reliability.Input.input["elimination"] = "yes"
        mechanism_reliability.Input.input["pf_elim"] = self.measure_p_solution
        mechanism_reliability.Input.input["pf_with_elim"] = np.min(
            [self.measure_pf_solution, 1.0e-16]
        )
