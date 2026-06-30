# pseudoavalanche

A Python package to study many-body localization thermalization avalanches with engineered baths.

## Install

```bash
python -m pip install -e .
```

Optional Clifford support:

```bash
python -m pip install -e .[qiskit]
```

## Quick route

```bash
python scripts/make_grid.py
bash scripts/job.slurm.sh
python scripts/collect.py
```
