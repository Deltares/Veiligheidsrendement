import numpy as np

__model_factor: float = 1.06


def calculate_reliability(safety_factor: np.ndarray) -> float:
    """
    Calculates the reliability (beta) based on the input arguments.
    Args:
        safety_factor (float): the safety factor to calculate the reliability with.
    Returns:
        float: the safety factor.
    """
    beta = ((safety_factor / __model_factor) - 0.41) / 0.15
    # Replace values greater than 8.0 with 8.0
    beta[beta > 8.0] = 8.0
    return beta


def calculate_safety_factor(reliability: float) -> float:
    """
    Calculates the safety factor based on the input arguments.
    Args:
        reliability (float): the reliability (beta) to calculate the safety factor with.
    Returns:
        float: the safety factor.
    """
    return (0.41 + 0.15 * reliability) * __model_factor
