# from vrtool.failure_mechanisms.revetment.slope_part import SlopePart


class RevetmentDataClass:
    def __init__(self):
        self.slope_parts = []  # for a list of SlopePart
        self.grass_relations = []  # for a list of RelationGrassRevetment
        self.stone_relations = []  # for a list of RelationStoneRevetment

    def current_transition_level(self) -> float:
        min_value_grass = 1e99
        for slope_part in self.slope_parts:
            if slope_part.is_grass:
                if slope_part.begin_part < min_value_grass:
                    min_value_grass = slope_part.begin_part

        if min_value_grass == 1e99:
            raise ValueError("No slope part with grass found")

        return min_value_grass
