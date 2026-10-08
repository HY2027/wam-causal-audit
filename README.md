# WAM Causal Audit

Method implementations and reproducibility materials for effect-grounded
auditing of architecture-defined action interfaces in world-action models.

The central distinction is between propagation-allowed node influence and
direct action dependence under strict consumer-source control. This repository
also provides propagated-future restoration and reduced-world-computation
analysis code. These quantities are not information shares or natural mediation
effects.

## Quick start: reproduce saved-result figures (CPU only)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-figures.txt
python -m pip install --no-deps -e .
python scripts/check_release.py
python -m unittest discover -s tests
python scripts/reproduce_figures.py
```

This reproduces three figures from existing saved plotting tables:

1. Physical response across model interfaces and registered factors/phases.
2. Joint strict attribution and pre-specified early/late restoration.
3. Reduced computation, matched responses and held-fold residuals.

PDF/PNG outputs are in `reference/workspace/FastWAM/figures/`.
No model, simulator, fitting, or new experiment is launched. Summary estimates
are not expanded into fabricated case-level samples.

## Contents

| Directory | Contents |
|---|---|
| `reference/workspace/` | Selected historical method, analysis and plotting implementations; original grouping retained |
| `reference/data/` | Selected reference runtime code and saved figure tables |
| `src/wam_causal_audit/` | Explicit path-relocation helpers |
| `scripts/` | CPU-only reproduction and static release checks |
| `tests/` | Packaging/path contract tests; not model validation |
| `provenance/` | Original/release hashes, export scope and check results |
| `docs/` | Method map, reproduction limits and external requirements |

See [the method map](docs/METHODS.md),
[reproduction scope](docs/REPRODUCIBILITY.md), and
[third-party notices](THIRD_PARTY_NOTICES.md).

## Scope of this release

Included: selected physical-donor operators, interface capture/replacement,
strict Joint current/future consumers, local propagated-future restoration,
shared-gain analysis, RoboTwin interface × command-state analysis, and
reduced-world-computation analysis.

Not bundled: checkpoints, raw observation/state/cache banks, full datasets,
videos, simulation assets, credentials, private logs, or unrelated attack and
training experiments. The code has not been revalidated by new model or simulator
runs during packaging. Historical runtime scripts still require their registered
assets and compatible deployment environments; they are not turnkey runners.

Existing frozen protocols and scientific results were not modified. Only release
copies were edited for path relocation and presentation. A public source release
does not turn exploratory/post-hoc studies into independent confirmation.

## Licensing

Existing FastWAM notices are retained in `LICENSES/`. A license decision for the
independently authored audit code is still pending; do not assume a blanket MIT
grant from the included upstream notice.
