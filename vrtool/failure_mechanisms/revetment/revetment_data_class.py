import numpy as np

GRASS_TYPE = 20.0


class SlopePart:
    # Stores data for slope part
    def __init__(
        self,
        begin_part: float,
        end_part: float,
        tan_alpha: float,
        top_layer_type: float,  # note that e.g. 26.1 is a valid type
        top_layer_thickness=np.NaN,
    ):
        self.begin_part = begin_part
        self.end_part = end_part
        self.tan_alpha = tan_alpha
        self.top_layer_type = top_layer_type
        self.top_layer_thickness = top_layer_thickness

    def is_grass(self) -> bool:
        return self.top_layer_type == GRASS_TYPE


class RelationGrassRevetment:
    # Stores data for relation grass revetment
    def __init__(self, year: int, transition_level: float, beta: float):
        self.year = year
        self.transition_level = transition_level
        self.beta = beta


class RelationStoneRevetment:
    # Stores data for relation stone revetment
    def __init__(
        self, slope_part: int, year: int, top_layer_thickness: float, beta: float
    ):
        self.year = year
        self.top_layer_thickness = top_layer_thickness
        self.beta = beta
        self.slope_part = slope_part


class RevetmentDataClass:
    def __init__(self):
        self.slope_parts = []  # for a list of SlopePart
        self.grass_relations = []  # for a list of RelationGrassRevetment
        self.stone_relations = []  # for a list of RelationStoneRevetment

    def current_transition_level(self) -> float:
        min_value_grass = 1e99
        for slope_part in self.slope_parts:
            if slope_part.is_grass():
                if slope_part.begin_part < min_value_grass:
                    min_value_grass = slope_part.begin_part

        if min_value_grass == 1e99:
            raise ValueError("No slope part with grass found")

        return min_value_grass
