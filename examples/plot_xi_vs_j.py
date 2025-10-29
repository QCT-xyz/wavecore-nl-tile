import os
import numpy as np
import matplotlib.pyplot as plt
from wavecore_nl.tile import Tile


def main():
    os.makedirs("artifacts", exist_ok=True)

    n = 12
    rng = np.random.default_rng(7)
    J = rng.integers(-1, 2, size=(n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)

    alpha = 0.3
    policies = ["fixed", "chirp", "alternating"]
    points = []

    for pol in policies:
        tile = Tile(modes=32, alpha=alpha)
        sched = tile.synthesize(J, policy=pol)
        res = tile.run(sched)
        points.append((pol, res.xi, res.j_score))

    # plot Ξ vs J
    plt.figure(figsize=(6, 4))
    for pol, xi, j in points:
        plt.scatter(xi, j, s=90)
        plt.text(xi, j, f"  {pol}", va="center", ha="left")

    plt.xlabel("Ξ (coherence)")
    plt.ylabel("J = E_cut − α·Ξ")
    plt.title("Policy sweep: Ξ vs J (α = 0.3)")
    plt.grid(True)
    out = "artifacts/xi_vs_j.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    for pol, xi, j in points:
        print(f"{pol:12s} | xi={xi:.6f}  J={j:.6f}")


if __name__ == "__main__":
    main()
