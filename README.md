# SETA-FL CIFAR-10 research scaffold

This repository is an **experimental scaffold**, not a validated SETA-FL implementation or a reproduction of published baselines.

## Contents

- `run_cifar_sequential.py`: minimal CIFAR-10 federated-training plumbing, Dirichlet partitioning, FedAvg aggregation, sign-flip attack option, and a prototype mixture-e-process diagnostic.
- `requirements.txt`: Python dependencies.
- `config.example.json`: suggested pilot configuration.
- `.gitignore`: excludes datasets, outputs, model checkpoints, and Python cache files.

## Important limitations (must be resolved before research claims)

1. The detector score is a clipped update-norm diagnostic, **not** a calibrated conditional residual. No formal false-alarm guarantee is established.
2. `--attack labelflip` is currently a placeholder and raises `NotImplementedError`; do not use it.
3. FLTrust, FoolsGold, JS-Drift, and other comparison algorithms are not implemented here.
4. The model is a small CNN, not ResNet-18.
5. The drift simulation swaps client partitions at the configured round; it is a controlled synthetic shift, not a natural temporal-drift benchmark.
6. The script currently uses sample-count-weighted FedAvg for aggregation; it does not yet use the sequential evidence to change aggregation weights.
7. Run smoke tests and verify dataset access before launching multi-seed experiments.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Smoke run

```bash
python run_cifar_sequential.py --clients 5 --rounds 2 --alpha 0.5 --attack none --device cpu
```

CIFAR-10 is downloaded by torchvision on first run, so internet access is required. If download is unavailable, place the dataset in `./data` using the torchvision CIFAR-10 layout.

## Pilot run

```bash
python run_cifar_sequential.py --seed 1 --alpha 0.5 --clients 10 --rounds 30 --attack signflip --device cpu
```

Outputs are written to `./results/` by default.

## Research integrity

Do not report the prototype e-process as statistically valid until its predictable score satisfies the required conditional null assumptions and calibration has been tested. Do not claim baseline comparisons until the baselines are independently implemented and run under matched settings.
