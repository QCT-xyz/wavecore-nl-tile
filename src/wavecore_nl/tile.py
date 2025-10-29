from dataclasses import dataclass
import numpy as np
from .config import settings
from .coherence import xi_from_phasors


@dataclass
class Schedule:
    phases: np.ndarray  # (depth, modes)
    pump: np.ndarray  # (depth,)
    coupling: np.ndarray  # (modes, modes)


@dataclass
class RunResult:
    spectrum: np.ndarray
    phase_noise: np.ndarray
    cut_value: float
    spins: np.ndarray
    xi: float


class Tile:
    def __init__(self, modes: int | None = None):
        self.modes = modes or settings.modes
        self.depth = settings.depth

    def synthesize(
        self, J: np.ndarray, alpha: float | None = None, policy: str = "chirp"
    ) -> Schedule:
        depth = self.depth
        modes = self.modes
        phases = np.linspace(0, np.pi, depth)[:, None] * np.ones((depth, modes))
        pump = np.linspace(0.1, 1.0, depth)
        coupling = np.zeros((modes, modes))
        n = min(J.shape[0], modes)
        scale = np.max(np.abs(J)) + 1e-9
        coupling[:n, :n] = J[:n, :n] / scale
        return Schedule(phases=phases, pump=pump, coupling=coupling)

    def run(self, sched: Schedule) -> RunResult:
        last_phase = sched.phases[-1]
        spins = np.where(np.sin(last_phase) >= 0, 1, -1)
        J = sched.coupling
        tri = np.triu_indices(J.shape[0], 1)
        sprod = np.outer(spins, spins)
        cut_value = float(np.sum(0.5 * (1.0 - sprod[tri]) * np.abs(J[tri])))

        phasors = np.exp(1j * last_phase)
        xi = xi_from_phasors(phasors)

        spectrum = np.abs(np.fft.rfft(last_phase))
        phase_noise = np.var(last_phase) * np.ones_like(spectrum)
        return RunResult(
            spectrum=spectrum, phase_noise=phase_noise, cut_value=cut_value, spins=spins, xi=xi
        )
