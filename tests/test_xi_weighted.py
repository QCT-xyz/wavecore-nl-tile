import numpy as np
from wavecore_nl.tile import Tile


def test_j_score_relation():
    n = 8
    rng = np.random.default_rng(123)
    J = rng.integers(-1, 2, size=(n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    alpha = 0.3
    tile = Tile(modes=16, alpha=alpha)
    sched = tile.synthesize(J)
    res = tile.run(sched)
    assert abs(res.j_score - (res.cut_value - alpha * res.xi)) < 1e-9
