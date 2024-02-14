import numpy as np

__model_factor: float = 1.06
BETA_THRESHOLD: float = 8.0


def calculate_reliability(safety_factor: np.ndarray) -> np.ndarray:
    """
    Calculates the reliability (beta) based on the input arguments.
    Args:
        safety_factor (float): the safety factor to calculate the reliability with.
    Returns:
        np.ndarray: the beta value(s) representing the reliability of the provided safety factor.
    """
    beta = ((safety_factor / __model_factor) - 0.41) / 0.15
    # Replace values greater than the threshold with the threshold itself.
    beta[beta > BETA_THRESHOLD] = BETA_THRESHOLD
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
