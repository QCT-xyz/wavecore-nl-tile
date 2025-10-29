import numpy as np
from wavecore_nl.tile import Tile


def test_tile_pipeline():
    n = 6
    rng = np.random.default_rng(1)
    J = rng.integers(-1, 2, size=(n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)

    tile = Tile(modes=16)
    sched = tile.synthesize(J)
    res = tile.run(sched)

    assert 0 <= res.xi <= 1, "Ξ must be within [0,1]"
    assert res.cut_value >= 0, "Cut value must be non-negative"
    assert res.spectrum.size > 0
