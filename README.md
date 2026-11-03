# SETA-FL: conditional prediction and sequential evidence scaffold

This is a development and mathematical-validation scaffold, **not** a validated
SETA-FL reproduction. The detector's theorem requires a correct predictable
conditional-mean envelope. Fitting history alone does not establish that envelope.
Aggregation is sample weighted subject to explicit client state policies. Unknown
clients are deferred by default; capped weights or predeclared clean warmup are
alternatives. `MONITORED` does not mean certified normal. This is an experimental
pipeline, not a validated complete method.

## Files

- `run_cifar_sequential.py`: SmallCNN, safe partial-participation FedAvg, seeded
  label/sign flipping, independent client histories and per-client monitoring.
- `seta_statistics.py`: trusted/quarantine buffers, frozen directional predictions,
  and a mixture of cumulative bounded-mean e-processes.
- `simulate_statistics.py`: reproducible scalar statistical simulations.
- `tests/`: training, attack, aggregation, predictability and mathematical tests.
- [docs/MATH.md](docs/MATH.md): old/new formulas, hypotheses, proof and limitations.
- [reports/STAGE2_REPORT.zh-CN.md](reports/STAGE2_REPORT.zh-CN.md): actual test results.
- [reports/stage3/STAGE3_REPORT.zh-CN.md](reports/stage3/STAGE3_REPORT.zh-CN.md):
  10,000-replication stress tests, calibration strategies, matched-data ablations,
  plots and negative results.
- [docs/STAGE4_ALIGNMENT.md](docs/STAGE4_ALIGNMENT.md): complete theory-to-code
  variable mapping, context differences, states and experiment-family error budget.
- `client_control.py`: predictable contextual regression, independent client states,
  reserved calibration/sequential budgets and server-retaining capped weights.
- `pilot_cifar.py`: gated real-data pilot with an immutable 36-run registry.
- `config.example.json`: pilot suggestions only; not automatically loaded.

## Environment and validation

Python 3.10 or 3.11 recommended. In the prepared cloud environment, use
`source .venv/bin/activate`. For a fresh CPU setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python simulate_statistics.py --trials 1000 --seed 20261103 --out reports
```

CUDA users need a matching official PyTorch CUDA distribution for their hardware,
rather than the CPU wheel command. GPU execution is covered by a conditional test;
a skipped test is not GPU validation. Set `MPLCONFIGDIR` to a writable directory if
the cloud HOME is read-only. No new dependency was added for these tests.

The simulations run 1000 independent replications in **each of seven scenarios**,
240 observations per replication. They do not train on CIFAR-10. The included
CSV has one row per replication; JSON includes seed, rates, Wilson intervals,
conditional mean delays and censored/restricted delay summaries.

## Stage-three statistical validation

See [docs/STAGE3_THEORY.md](docs/STAGE3_THEORY.md) for filtration, Hoeffding
parameters, the supermartingale proof and calibration failure budgets.
Fixed-anchor and rolling independent-reference bounds require a stable affine
conditional mean. The conservative envelope requires no mean model and has no
useful detection power. These certificates do not apply automatically to real FL.

The protocols in `configs/stage3*.json` were saved before their corresponding
runs. To reproduce statistical outputs, without CIFAR training:

```bash
python -m unittest discover -s tests -v
python stress_validation.py
python stress_validation.py --config configs/stage3_misspecification.json --out reports/stage3/misspecification
python stress_validation.py --config configs/stage3_boundary.json --out reports/stage3/boundary
python matched_ablation.py
export XDG_CACHE_HOME=/workspace/.cache
export MPLCONFIGDIR=/workspace/.cache/matplotlib
python plot_stage3.py
python build_stage3_report.py
python audit_stage3.py
```

Raw CSV files are gzip compressed. Every scenario/strength cell has 10,000
replications paired across three alpha levels and seven ablations. Predictor-only
methods have no detection rates. Each experiment preserves the runner source used
in `source_snapshot/`; its config and output hashes are recorded. Negative controls
explicitly demonstrate false alarms on unobserved benign drift and failures on
weak/late attacks. Do not tune parameters from these test outcomes.

## Dataset and training (real data blocked in this stage)

CIFAR-10 is loaded through torchvision with its standard integrity checks.
Downloads require `www.cs.toronto.edu` and its redirect
`cave.cs.toronto.edu`; cloud network settings may need review.
See [docs/CIFAR_IMPORT.md](docs/CIFAR_IMPORT.md) for integrity-checked local archive
or batch-directory import. `python cifar_smoke.py --root ./data` uses only 32 real
training images and one aggregation when data exists; otherwise it reports a
blocked result and does not substitute fake data.

For the stage-four pilot use `python pilot_cifar.py --root ./data --download`
(or omit `--download` after official local import). A fresh checksum-verified
32-image test must pass first. The predeclared protocol uses 10 clients, 20 rounds,
3 seeds, IID/Dirichlet, no attack/Sign Flipping and three admission policies. It
uses 2000 train and 1000 test images; this is a small-subset pilot, not a full-data
benchmark. The current official proxy response is 403, so no real pilot metrics
exist. See `reports/stage4/pilot_status.json` and `run_registry.csv`.

The ordinary CLI now requires `context_value` in external history records. Its
default `defer` policy does not aggregate clients with no references. To study
assumed attack-free warmup in a controlled experiment, explicitly use
`--uncalibrated-policy clean-warmup --declare-clean-warmup --warmup-rounds 8`
with more than eight rounds and attacks starting later. This is a protocol
assumption, not inferred normality. The stage-four runner supplies these settings.
`--bound-mode assumed` is uncertified; `conservative` uses a valid but powerless
universal envelope. No claim of real conditional drift adaptation is established.
A round includes all selected clients' local data unless dataset limits are set.
Use the gated pilot entry point for real experiments and unit tests for code checks.
Nonempty clients are eligible, `ceil(participation * eligible)` are sampled before
training, and sample-count weights are normalized over actual participants.
Empty clients never construct a shuffled loader; an all-empty round is a no-op.
Attacker IDs are fixed before participation selection and are independent of the
separately configurable drift start. Label flips are seeded class permutations
with no fixed labels and do not mutate the underlying dataset.

## Trusted history and prediction

The detector is disabled for a client until it has at least eight independently
validated or explicitly protocol-declared reference vectors. No monitoring update,
even a non-alarm, enters normal history automatically. Current observations are
quarantined; declared warmup admission is a separate explicit operation.
Provide `--trusted-history path.json` only from an externally validated source:

```json
{
  "0": [
    {"features": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     "context_value": 0.0,
     "provenance": "independent clean validation record identifier"}
  ]
}
```

The example shows the schema, not sufficient history. Each vector has 16
coordinates, norm at most one, and must use the same feature transform and
`--feature-scale` as the monitored updates. Provenance text is not authentication.
All client predictions are frozen before current-round training.

`--mean-error-bound` is an **assumed** upper bound on conditional mean prediction
error. Its default is not a statistical certificate. Without a justified bound,
there is no strict real-FL false-alarm guarantee. `--family-alpha` is split across
the fixed client set, independent scalar tests and predeclared run family, with
separate sequential and calibration shares; it is not reallocated as participation changes. The CLI
`--alpha` remains Dirichlet concentration, not the test significance level.
The program logs whether each detector is enabled, its score/prediction/evidence,
trusted count, actual weight, and latched alarm to `results/`.

## Remaining research work

- Validate conditional prediction under model evolution, Non-IID data, participation
  selection and benign drift; implement an independently justified error envelope.
- Analyze the candidate alarm filter's selection feedback, convergence and calibration.
- Evaluate real-data detection and implement reference baselines under matched settings.
- Improve feature coverage and late/intermittent attack power; the low-dimensional
  sketch can miss attacks.
- Add externally authenticated clean-history admission and recovery/checkpoint policy.

The simulations intentionally demonstrate that **unmodeled normal drift can cause
false alarms**. Do not claim arbitrary-drift robustness, universal attack detection,
or a strict real-data false-alarm guarantee from these results.
