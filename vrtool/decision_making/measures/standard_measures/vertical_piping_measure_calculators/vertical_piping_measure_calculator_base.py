import copy
from abc import ABC, abstractmethod

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class VerticalPipingMeasureCalculatorBase(ABC):
    """
    Abstract class to represent the basic inheritance of all the `VerticalPipingMeasureCalculatorProtocol`
    concrete classes as, for now, they all share almost the same logic.

    __DO NOT__ start adding `if-else` statements based on reflection. When needed, please define other base classes or move
    the logic to the corresponding concrete class via inheritance and overriding of methods.
    """

    traject_info: DikeTrajectInfo
    dike_section: DikeSection
    reliability_years: list[int]
    computation_year_start: int
    measure_year: int

    @property
    @abstractmethod
    def pf_piping_reduction_factor(self) -> float:
        """
        Gets the default reduction factor for `pf_piping` ( `P_solution` ).
        This property can be overriden when inheriting from this class.

        Returns:
            float: reduction value.
        """

    @classmethod
    def from_measure_section_traject(
        cls,
        measure: MeasureProtocol,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
    ):
        """
        Initializes a concrete instance of `VerticalPipingMeasureCalculatorBase`
        with its arguments already

        Args:
            measure (MeasureProtocol): Measure to be applied to the dike's section.
            dike_section (DikeSection):  Dike section properties.
            traject_info (DikeTrajectInfo): Dike traject in which the dike section takes place.

        Returns:
            VerticalPipingMeasureCalculatorBase: Concrete instance of a `VerticalPipingMeasureCalculatorBase`.
        """
        _calculator = cls()
        _calculator.traject_info = traject_info
        _calculator.dike_section = dike_section
        _calculator.reliability_years = measure.config.T
        _calculator.computation_year_start = measure.config.t_0
        _calculator.measure_year = measure.parameters["year"]
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

    def _copy_results(
        self, target: MechanismReliability, source_input: MechanismReliability
    ) -> None:
        target.Input = copy.deepcopy(source_input.Input)

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

    def _configure_piping(
        self,
        mechanism_reliability: MechanismReliability,
        year_to_calculate: str,
        dike_section_piping_reliability: MechanismReliability,
    ) -> None:
        if int(year_to_calculate) < self.measure_year:
            self._copy_results(mechanism_reliability, dike_section_piping_reliability)

        mechanism_reliability.Input.input["elimination"] = "yes"
        mechanism_reliability.Input.input[
            "piping_reduction_factor"
        ] = self.pf_piping_reduction_factor
