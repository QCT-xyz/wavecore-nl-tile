import numpy as np
from wavecore_nl.tile import Tile
from wavecore_nl.onnx_export import export_stub


def main():
    n = 8
    rng = np.random.default_rng(0)
    J = rng.integers(-1, 2, size=(n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)

    tile = Tile(modes=16)
    sched = tile.synthesize(J)
    res = tile.run(sched)
    print(f"cut_value: {res.cut_value:.4f}  xi: {res.xi:.6f}")
    path = export_stub(sched)
    print(f"exported: {path}")


if __name__ == "__main__":
    main()
