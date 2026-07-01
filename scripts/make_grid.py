import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pseudoavalanche.config import AvalancheConfig


def main() -> None:
    base = AvalancheConfig(
        N0=4,
        L=3,
        xi=2.0,
        g=1.0,
        h_dist=("uniform", -0.5, 0.5),
        J=0.0,
        lambda_dope=0,
        spec_mode="gue",
        W=1.0,
        dt=0.1,
        t_max=2.0,
        n_typicality=2,
        thermalization_threshold=0.4,
    )
    grid = []
    xi_values = [2.0, 3.0]
    g_values = [1.0, 2.0]
    lambda_values = [0, "haar"]
    spec_modes = ["gue", "poisson"]
    seeds = list(range(1, 5))

    for xi in xi_values:
        for g in g_values:
            for lambda_dope in lambda_values:
                for spec_mode in spec_modes:
                    config = AvalancheConfig(
                        **{
                            **base.__dict__,
                            "xi": xi,
                            "g": g,
                            "lambda_dope": lambda_dope,
                            "spec_mode": spec_mode,
                            "t_max": 2.0,
                        }
                    )
                    for seed in seeds:
                        grid.append({"config": config.__dict__, "seed": seed})

    out_path = Path(__file__).resolve().parent / "job_list.json"
    out_path.write_text(json.dumps(grid, indent=2))
    print("Recommended sweep regime:")
    print(f"  xi: {xi_values}")
    print(f"  g: {g_values}")
    print(f"  t_max: 2.0")
    print(f"  lambda_dope: {lambda_values}")
    print(f"  spec_mode: {spec_modes}")
    print(f"  thermalization_threshold: 0.4")
    print(f"  seeds: {seeds}")
    print(f"Wrote {len(grid)} jobs to {out_path}")


if __name__ == "__main__":
    main()
