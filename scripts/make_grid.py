import json
from pathlib import Path

from pseudoavalanche.config import AvalancheConfig


def main() -> None:
    base = AvalancheConfig(
        N0=4,
        L=3,
        xi=2.0,
        g=0.5,
        h_dist=("uniform", -0.5, 0.5),
        J=0.0,
        lambda_dope=0,
        spec_mode="gue",
        W=1.0,
        dt=0.1,
        t_max=0.5,
        n_typicality=2,
    )
    grid = []
    xi_values = [1.0, 2.0]
    g_values = [0.5, 1.0]
    lambda_values = [0, "haar"]
    spec_modes = ["gue", "poisson"]
    seeds = list(range(1, 5))

    for xi in xi_values:
        for g in g_values:
            for lambda_dope in lambda_values:
                for spec_mode in spec_modes:
                    config = AvalancheConfig(**{**base.__dict__, "xi": xi, "g": g, "lambda_dope": lambda_dope, "spec_mode": spec_mode})
                    for seed in seeds:
                        grid.append({"config": config.__dict__, "seed": seed})

    out_path = Path(__file__).resolve().parent / "job_list.json"
    out_path.write_text(json.dumps(grid, indent=2))
    print(f"Wrote {len(grid)} jobs to {out_path}")


if __name__ == "__main__":
    main()
