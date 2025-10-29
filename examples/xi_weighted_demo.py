import numpy as np
from wavecore_nl.tile import Tile

if __name__ == "__main__":
    n = 8
    rng = np.random.default_rng(42)
    J = rng.integers(-1, 2, size=(n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)

    tile = Tile(modes=16, alpha=0.3)
    sched = tile.synthesize(J)
    res = tile.run(sched)

    print(f"cut={res.cut_value:.4f}")
    print(f"xi={res.xi:.6f}")
    print(f"J={res.j_score:.6f} (α={tile.alpha})")
