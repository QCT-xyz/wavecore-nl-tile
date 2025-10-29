import numpy as np


def xi_from_phasors(phasors: np.ndarray) -> float:
    """
    Simple coherence proxy in [0,1]:
    |sum e^{iθ_k}| / sum |e^{iθ_k}|
    """
    if phasors.size == 0:
        return 0.0
    return float(np.abs(np.sum(phasors)) / np.sum(np.abs(phasors)))
