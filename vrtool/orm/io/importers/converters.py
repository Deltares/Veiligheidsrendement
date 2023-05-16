from typing import Union


def to_valid_float(float_value: Union[float, None]) -> float:
    """
    Converts an ORM FloatField value to python float. This prevents None values from being set instead of NaNs.
    To be used when the definition of a `OrmBaseModel` has FloatField(null=True).

    Args:
        float_value (Union[float, None]): Value that could be None or already float.

    Returns:
        float: Valid float value.
    """
    if not float_value:
        return float("nan")
    return float_value