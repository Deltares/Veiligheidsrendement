class RelationStoneRevetment:
    # Stores data for relation stone revetment
    def __init__(
        self, slope_part: int, year: int, top_layer_thickness: float, beta: float
    ):
        self.year = year
        self.top_layer_thickness = top_layer_thickness
        self.beta = beta
        self.slope_part = slope_part
