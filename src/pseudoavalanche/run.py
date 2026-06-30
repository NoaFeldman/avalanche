from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .analysis import fit_v_log
from .bath import Bath
from .config import AvalancheConfig
from .evolve import evolve
from .observables import compute_observables
from .state import initial_state


def run_job(config: AvalancheConfig, seed: int) -> dict:
    bath = Bath(config, seed=seed)
    psi0 = initial_state(config, seed=seed + 1)
    times, trajectories = evolve(config, psi0, bath)
    obs = compute_observables(times, trajectories, config)
    return {
        "times": times,
        "front": obs["front"],
        "t_ell": obs["t_ell"],
        "s": obs["s"],
        "entropies": obs["entropies"],
        "v_log": fit_v_log(times, obs["front"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one pseudoavalanche job.")
    parser.add_argument("--job-list", type=Path, required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    with args.job_list.open("r", encoding="utf-8") as handle:
        jobs = json.load(handle)

    if args.job_index < 1 or args.job_index > len(jobs):
        raise ValueError("job-index must be between 1 and number of jobs")
    job = jobs[args.job_index - 1]
    config = AvalancheConfig(**job["config"], seed=int(job["seed"]))
    result = run_job(config, config.seed)

    output_path = args.output_dir / config.param_hash
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path / f"seed_{config.seed}.npz", **result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
