GRASS_TYPE = 20.0


class SlopePart:
    # Stores data for slope part
    def __init__(
        self,
        begin_part: float,
        end_part: float,
        tan_alpha: float,
        top_layer_type: float,  # note that e.g. 26.1 is a valid type
        top_layer_thickness=float("nan"),
    ):
        self.begin_part = begin_part
        self.end_part = end_part
        self.tan_alpha = tan_alpha
        self.top_layer_type = top_layer_type
        self.top_layer_thickness = top_layer_thickness

    def is_grass(self) -> bool:
        return self.top_layer_type == GRASS_TYPE
