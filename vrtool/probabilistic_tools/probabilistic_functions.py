from scipy.stats import norm


def beta_to_pf(beta: float | list[float]) -> float | list[float]:
    # alternative: use scipy
    if isinstance(beta, list):
        return norm.cdf([-_element for _element in beta]).tolist()
    return norm.cdf(-beta)


def pf_to_beta(pf):
    # alternative: use scipy
    return -norm.ppf(pf)
