# WAM Causal Audit

Code and reproducibility materials for causal auditing of action-facing
interfaces in world-action models, including source conditioning, local
restoration, and reduced predictive computation.

## Quick start

Reproduce the paper figures from saved results:

```bash
python -m pip install -r requirements-figures.txt
python -m pip install --no-deps -e .
python scripts/reproduce_figures.py
```

PDF/PNG outputs: `reference/workspace/FastWAM/figures/`.
Full experiments require external model weights and benchmark assets.

See [Methods](docs/METHODS.md), [Reproducibility](docs/REPRODUCIBILITY.md),
and [Third-party notices](THIRD_PARTY_NOTICES.md) for details.
