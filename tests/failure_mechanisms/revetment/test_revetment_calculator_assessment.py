import json

import pytest

from tests import test_data
from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_calculator import RevetmentCalculator
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.revetment_calculation_assessment import (
    RevetmentCalculation,
)
from tests import test_data
from vrtool.failure_mechanisms.revetment.slope_part import (
    StoneSlopePart,
    SlopePartProtocol,
    SlopePartBuilder,
)


class TestRevetmentCalculatorAssessment:
    def _read_JSON(self, file_name):
        with open(file_name, "r") as openfile:
            json_object = json.load(openfile)
        return json_object

    def searchSlopePart(
        self, slope_parts: list[SlopePartProtocol], slope_part: SlopePartProtocol
    ) -> tuple[bool, SlopePartProtocol]:
        for part in slope_parts:
            if part.begin_part == slope_part.begin_part:
                return [True, part]
        return [False, slope_part]

    def _convertJsonObjects(
        self, dataZST, dataGEBU, revetment: RevetmentDataClass
    ) -> RevetmentDataClass:
        n_sections = dataZST["aantal deelvakken"]
        for _n_section in range(n_sections):
            _slope_part = SlopePartBuilder.build(
                top_layer_type=dataZST["toplaagtype"][_n_section],
                begin_part=dataZST["Zo"][_n_section],
                end_part=dataZST["Zb"][_n_section],
                tan_alpha=dataZST["tana"][_n_section],
                top_layer_thickness=dataZST["D huidig"][_n_section],
            )

            [exists, slope] = self.searchSlopePart(revetment.slope_parts, _slope_part)
            if not exists:
                revetment.slope_parts.append(_slope_part)

            if isinstance(slope, StoneSlopePart):
                key = f"deelvak {_n_section}"
                nBeta = len(dataZST[key]["betaFalen"])
                for m in range(nBeta):
                    rel = RelationStoneRevetment(
                        dataZST["zichtjaar"],
                        dataZST[key]["D_opt"][m],
                        dataZST[key]["betaFalen"][m],
                    )
                    slope.slope_part_relations.append(rel)

        nGrass = len(dataGEBU["grasbekleding_begin"])
        for _n_section in range(nGrass):
            rel = RelationGrassRevetment(
                dataGEBU["zichtjaar"],
                dataGEBU["grasbekleding_begin"][_n_section],
                dataGEBU["betaFalen"][_n_section],
            )
            revetment.grass_relations.append(rel)

        return revetment

    def _get_revetment_input(
        self, years: list[int], section: int
    ) -> RevetmentDataClass:
        revetment = RevetmentDataClass()
        for year in years:
            gebuFile = f"revetment/GEBU_{section}_{year}.json"
            dataGEBU = self._read_JSON(test_data / gebuFile)
            zstFile = f"revetment/ZST_{section}_{year}.json"
            dataZST = self._read_JSON(test_data / zstFile)
            revetment = self._convertJsonObjects(dataZST, dataGEBU, revetment)
        return revetment

    @pytest.mark.parametrize(
        "assessment_year, given_years, section_id, ref_values",
        [
            pytest.param(
                2025,
                [2025],
                0,
                [3.6112402089287357, 0.00015236812335053454],
                id="2025_0",
            ),
            pytest.param(
                2100,
                [2100],
                0,
                [3.617047156851664, 0.00014899151576941146],
                id="2100_0",
            ),
            pytest.param(
                2050,
                [2025, 2100],
                0,
                [3.6131758582363784, 0.0001512347051563111],
                id="2050_0",
            ),
        ],
    )
    def test_revetment_calculation(
        self,
        assessment_year: int,
        given_years: list[int],
        section_id: int,
        ref_values: list[float],
    ):
        revetment = self._get_revetment_input(given_years, section_id)

        calc = RevetmentCalculator(revetment)
        [beta, pf] = calc.calculate(assessment_year)

        assert beta == pytest.approx(ref_values[0], rel=1e-8)
        assert pf == pytest.approx(ref_values[1], 1e-8)
