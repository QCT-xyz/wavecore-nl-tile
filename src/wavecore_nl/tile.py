from dataclasses import dataclass
import numpy as np
from .config import settings
from .coherence import xi_over_depth


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
    j_score: float  # J = E_cut - alpha * xi


class Tile:
    def __init__(self, modes: int | None = None, alpha: float | None = None):
        self.modes = modes or settings.modes
        self.depth = settings.depth
        self.alpha = settings.alpha if alpha is None else alpha

    def synthesize(
        self, J: np.ndarray, alpha: float | None = None, policy: str = "chirp"
    ) -> Schedule:
        if alpha is not None:
            self.alpha = float(alpha)
        depth = self.depth
        modes = self.modes

        # --- schedule policies ---
        if policy == "fixed":
            # Constant phase sheet, steady pump -> high Ξ, minimal cut dynamics
            phases = np.zeros((depth, modes))
            pump = np.full(depth, 0.7)
        elif policy == "alternating":
            # Alternate phase windows and pump bursts -> lower Ξ, more cut structure
            phases = np.zeros((depth, modes))
            for d in range(depth):
                if d % 2 == 0:
                    phases[d] = np.linspace(0.0, np.pi, modes)
                else:
                    phases[d] = np.linspace(np.pi, 2.0 * np.pi, modes)
            pump = np.tile([0.35, 0.95], (depth // 2 + 1))[:depth]
        else:
            # "chirp" (default): gentle time chirp + per-mode offset -> mid Ξ
            base = np.linspace(0.0, np.pi, depth)[:, None]  # time ramp
            per_mode = np.linspace(0.0, np.pi / 4.0, modes)[None, :]  # mode-dependent offset
            phases = (base + per_mode) % (2.0 * np.pi)
            pump = np.linspace(0.2, 1.0, depth)

        coupling = np.zeros((modes, modes))
        n = min(J.shape[0], modes)
        scale = np.max(np.abs(J)) + 1e-9
        coupling[:n, :n] = J[:n, :n] / scale
        return Schedule(phases=phases, pump=pump, coupling=coupling)

    def run(self, sched: Schedule) -> RunResult:
        last_phase = sched.phases[-1]
        spins = np.where(np.sin(last_phase) >= 0, 1, -1)
        Jmat = sched.coupling
        tri = np.triu_indices(Jmat.shape[0], 1)
        sprod = np.outer(spins, spins)
        cut_value = float(np.sum(0.5 * (1.0 - sprod[tri]) * np.abs(Jmat[tri])))

        # depth-aware coherence over the whole schedule
        xi = xi_over_depth(sched.phases)

        spectrum = np.abs(np.fft.rfft(last_phase))
        phase_noise = np.var(last_phase) * np.ones_like(spectrum)

        j_score = cut_value - self.alpha * xi
        return RunResult(
            spectrum=spectrum,
            phase_noise=phase_noise,
            cut_value=cut_value,
            spins=spins,
            xi=xi,
            j_score=j_score,
        )
