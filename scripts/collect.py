import csv
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "results"
    rows = []
    for param_dir in sorted(root.glob("*")):
        if not param_dir.is_dir():
            continue
        front_vals = []
        v_logs = []
        for seed_file in sorted(param_dir.glob("seed_*.npz")):
            data = np.load(seed_file)
            v_logs.append(float(data["v_log"]))
            front_vals.append(data["front"])
        if not v_logs:
            continue
        summary = {
            "param_hash": param_dir.name,
            "n_seeds": len(v_logs),
            "v_log_mean": float(np.mean(v_logs)),
            "v_log_std": float(np.std(v_logs, ddof=1)) if len(v_logs) > 1 else 0.0,
        }
        rows.append(summary)
    out_dir = root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "summary.npz", **{r["param_hash"]: r for r in rows})
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["param_hash", "n_seeds", "v_log_mean", "v_log_std"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary for {len(rows)} parameter sets to {out_dir}")


if __name__ == "__main__":
    main()
