# SETA-FL: conditional prediction and sequential evidence scaffold

This is a development and mathematical-validation scaffold, **not** a validated
SETA-FL reproduction. The detector's theorem requires a correct predictable
conditional-mean envelope. Fitting history alone does not establish that envelope.
Aggregation remains sample-weighted FedAvg; detection does not yet set trust weights.

## Files

- `run_cifar_sequential.py`: SmallCNN, safe partial-participation FedAvg, seeded
  label/sign flipping, independent client histories and per-client monitoring.
- `seta_statistics.py`: trusted/quarantine buffers, frozen directional predictions,
  and a mixture of cumulative bounded-mean e-processes.
- `simulate_statistics.py`: reproducible scalar statistical simulations.
- `tests/`: training, attack, aggregation, predictability and mathematical tests.
- [docs/MATH.md](docs/MATH.md): old/new formulas, hypotheses, proof and limitations.
- [reports/STAGE2_REPORT.zh-CN.md](reports/STAGE2_REPORT.zh-CN.md): actual test results.
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

## Dataset and training (not run in this stage)

CIFAR-10 is loaded through torchvision with its standard integrity checks.
Downloads require `www.cs.toronto.edu` and its redirect
`cave.cs.toronto.edu`; cloud network settings may need review.
For a later authorized run, the CLI supports:

```bash
python run_cifar_sequential.py --clients 5 --rounds 2 --attack none --device cpu
# Example of a future intermittent label-flip run:
python run_cifar_sequential.py --clients 5 --rounds 2 --attack labelflip \
  --attack-start 1 --attack-mode intermittent --attack-period 4 \
  --attack-duration 2 --participation 0.5 --device cpu
```

A round includes all selected clients' local data. These commands are not a
tiny-data smoke test. Use the unit tests for the current no-full-training stage.
Nonempty clients are eligible, `ceil(participation * eligible)` are sampled before
training, and sample-count weights are normalized over actual participants.
Empty clients never construct a shuffled loader; an all-empty round is a no-op.
Attacker IDs are fixed before participation selection and are independent of the
separately configurable drift start. Label flips are seeded class permutations
with no fixed labels and do not mutate the underlying dataset.

## Trusted history and prediction

The detector is disabled for a client until it has at least eight independently
validated normal feature vectors. No current update, even a non-alarm, enters
normal history automatically. Current observations are quarantined.
Provide `--trusted-history path.json` only from an externally validated source:

```json
{
  "0": [
    {"features": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
the fixed client set, not reallocated as participation changes. The CLI
`--alpha` remains Dirichlet concentration, not the test significance level.
The program logs whether each detector is enabled, its score/prediction/evidence,
trusted count, actual weight, and latched alarm to `results/`.

## Remaining research work

- Validate conditional prediction under model evolution, Non-IID data, participation
  selection and benign drift; implement an independently justified error envelope.
- Define and analyze how sequential evidence changes aggregation weights.
- Evaluate real-data detection and implement reference baselines under matched settings.
- Improve feature coverage and late/intermittent attack power; the low-dimensional
  sketch can miss attacks.
- Add externally authenticated clean-history admission and recovery/checkpoint policy.

The simulations intentionally demonstrate that **unmodeled normal drift can cause
false alarms**. Do not claim arbitrary-drift robustness, universal attack detection,
or a strict real-data false-alarm guarantee from these results.
