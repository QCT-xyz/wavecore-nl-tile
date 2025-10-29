import numpy as np


def xi_from_phasors(phasors: np.ndarray) -> float:
    """
    Legacy single-snapshot coherence in [0,1]:
    |sum e^{iθ_k}| / sum |e^{iθ_k}|
    """
    if phasors.size == 0:
        return 0.0
    return float(np.abs(np.sum(phasors)) / np.sum(np.abs(phasors)))


def xi_over_depth(phases: np.ndarray) -> float:
    """
    Depth-aware coherence in [0,1] combining:
      1) Spatial coherence per depth: R_d = |mean_k e^{i θ_{d,k}}|
      2) Temporal smoothness per mode: S  = mean_k |mean_d e^{i Δθ_{d,k}}|
    Ξ = 0.7 * mean(R_d) + 0.3 * S
    """
    if phases.ndim != 2:
        return 0.0
    depth, modes = phases.shape
    if depth == 0 or modes == 0:
        return 0.0

    # spatial coherence per depth
    phasors = np.exp(1j * phases)  # (depth, modes)
    R_d = np.abs(np.mean(phasors, axis=1))  # (depth,)
    R_bar = float(np.mean(R_d))

    # temporal smoothness per mode (forward differences)
    if depth > 1:
        dtheta = np.diff(phases, axis=0)  # (depth-1, modes)
        s_ph = np.exp(1j * dtheta)  # (depth-1, modes)
        S_k = np.abs(np.mean(s_ph, axis=0))  # (modes,)
        S = float(np.mean(S_k))
    else:
        S = 1.0  # single layer is trivially smooth

    xi = 0.7 * R_bar + 0.3 * S
    if not np.isfinite(xi):
        return 0.0
    return float(np.clip(xi, 0.0, 1.0))
