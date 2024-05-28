import math


def floats_are_equal_or_nan(left_float: float, right_float: float) -> bool:
    """
    Compares two floats for equality, when both are `float("nan")` then
    we considered them as equal.
    """
    if math.isnan(left_float) and math.isnan(right_float):
        return True
    return left_float == right_float
