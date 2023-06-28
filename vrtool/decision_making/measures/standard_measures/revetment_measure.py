from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.failure_mechanisms.revetment.revetment_calculation_assessment import (
    RevetmentCalculation,
)
from vrtool.flood_defence_system.dike_section import DikeSection


class RevetmentMeasure(MeasureProtocol):
    def __init__(self, revetment_calculation: RevetmentCalculation) -> None:
        self.revetment_mechanism_calculator = revetment_calculation

    def _calculate_section_reliability(self) -> float:
        pass

    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool,
    ):
        self.measures = {}
        self.measures["Reliability"] = self._calculate_section_reliability()
        self.measures["Cost"] = self.revetment_mechanism_calculator.calculate_cost(
            dike_section.Length, year
        )

    def _evaluate_measure_karolina_script(self):
        measure = {
            "Zo": list(),
            "Zb": list(),
            "toplaagtype": list(),
            "D": list(),
            "betaZST": list(),
            "betaGEBU": list(),
            "previous toplaagtype": list(),
            "reinforce": list(),
            "tana": list(),
        }

        count = 0
        for i in range(0, dataZST["aantal deelvakken"]):

            if dataZST["Zb"][i] <= h_onder:

                count += 1
                # steen
                topd, betares, reinforce = design_steen(
                    beta,
                    dataZST[f"deelvak {i}"]["D_opt"],
                    dataZST[f"deelvak {i}"]["betaFalen"],
                    dataZST["toplaagtype"][i],
                    dataZST["D huidig"][i],
                )

                measure["Zo"].append(dataZST["Zo"][i])
                measure["Zb"].append(dataZST["Zb"][i])
                measure["toplaagtype"].append(dataZST["toplaagtype"][i])
                measure["previous toplaagtype"].append(dataZST["toplaagtype"][i])
                measure["D"].append(topd)
                measure["betaZST"].append(betares)
                measure["betaGEBU"].append(np.nan)
                measure["reinforce"].append(reinforce)
                measure["tana"].append(dataZST["tana"][i])

            elif (
                dataZST["Zo"][i] < h_onder and dataZST["Zb"][i] > h_onder
            ):  # part is steen and part is gras

                count += 2
                # steen
                topd, betares, reinforce = design_steen(
                    beta,
                    dataZST[f"deelvak {i}"]["D_opt"],
                    dataZST[f"deelvak {i}"]["betaFalen"],
                    dataZST["toplaagtype"][i],
                    dataZST["D huidig"][i],
                )

                measure["Zo"].append(dataZST["Zo"][i])
                measure["Zb"].append(h_onder)
                measure["toplaagtype"].append(dataZST["toplaagtype"][i])
                measure["previous toplaagtype"].append(dataZST["toplaagtype"][i])
                measure["D"].append(topd)
                measure["betaZST"].append(betares)
                measure["betaGEBU"].append(np.nan)
                measure["reinforce"].append(reinforce)
                measure["tana"].append(dataZST["tana"][i])

                # gras
                measure["Zo"].append(h_onder)
                measure["Zb"].append(dataZST["Zb"][i])
                measure["toplaagtype"].append(20.0)
                measure["previous toplaagtype"].append(dataZST["toplaagtype"][i])
                measure["D"].append(np.nan)
                measure["betaZST"].append(np.nan)
                measure["betaGEBU"].append(
                    evaluate_gras(
                        h_onder, dataGEBU["grasbekleding_begin"], dataGEBU["betaFalen"]
                    )
                )
                measure["reinforce"].append("yes")
                measure["tana"].append(dataZST["tana"][i])

            elif dataZST["Zo"][i] >= h_onder:  # gras

                count += 1
                # gras
                measure["Zo"].append(dataZST["Zo"][i])
                measure["Zb"].append(dataZST["Zb"][i])
                measure["toplaagtype"].append(20.0)
                measure["previous toplaagtype"].append(dataZST["toplaagtype"][i])
                measure["D"].append(np.nan)
                measure["betaZST"].append(np.nan)
                measure["betaGEBU"].append(
                    evaluate_gras(
                        h_onder, dataGEBU["grasbekleding_begin"], dataGEBU["betaFalen"]
                    )
                )
                measure["reinforce"].append("yes")
                measure["tana"].append(dataZST["tana"][i])

        if h_onder >= np.max(dataZST["Zb"]):

            if h_onder >= kruinhoogte:
                raise ValueError("Overgang >= kruinhoogte")

            count += 1
            # gras
            measure["Zo"].append(h_onder)
            measure["Zb"].append(kruinhoogte)
            measure["toplaagtype"].append(20.0)
            measure["previous toplaagtype"].append(np.nan)
            measure["D"].append(np.nan)
            measure["betaZST"].append(np.nan)
            measure["betaGEBU"].append(
                evaluate_gras(
                    h_onder, dataGEBU["grasbekleding_begin"], dataGEBU["betaFalen"]
                )
            )
            measure["reinforce"].append("yes")
            measure["tana"].append(dataZST["tana"][i])

        return count, measure
