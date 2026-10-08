# Reproduction scope and requirements

## Included, executable without a model

Run `python scripts/reproduce_figures.py` after installing the CPU dependencies.
It redraws the physical-response, Joint-attribution/restoration, and
reduced-computation figures from the included saved tables. It performs no
model inference, simulator operation, bootstrap, fitting, interpolation, or
synthetic sample generation.

Outputs are written under `reference/workspace/FastWAM/figures/`.
The tables include both published summary estimates and selected real
case/source-level plotting records. A summary row is never treated as a sample.
The 98 validation records used in the Joint figure and 20 restoration source
records retain their original selection; no new selection is performed.

The plotting scripts preserve the existing numeric hash/value guards. Only
path-text anonymization changes some CSV byte hashes; the corresponding public
copy's hash is used in the plotting guard and both hashes are listed in
`provenance/source_export.json`. Numerical CSV cells remain unchanged.

## Included for inspection; not end-to-end runtime validated in this release

The historical method implementations and their local Python dependencies are
in `reference/workspace/` and `reference/data/`. Their syntax is checked without
importing them. New models, inference, simulation, and scientific statistics
were not run while preparing this release.

The original workspace path literals have been replaced with explicit
`@WORKSPACE@` and `@DATA@` expressions resolved by `wam_causal_audit.paths`.
Defaults point into this release's `reference/` tree. Set `WAM_WORKSPACE` and
`WAM_DATA` only for a separate, deliberately prepared execution environment.
Paths inside externally supplied manifests are not silently rewritten. Relative
filenames and content hashes inside those manifests must be audited separately.

## External requirements not bundled

- Deployment code and compatible environments for FastWAM/legacy `attackwam`,
  ImageWAM, LIBERO and RoboTwin. They have separate licenses and dependencies.
- Checkpoints, VAEs/text encoders, processor/config files, normalization statistics,
  robot assets, task datasets and registered observation/state packages.
- Full source registries, runtime manifests, raw action/cache banks, and the
  original provenance/hash chain for each complete experiment.
- External legacy request files referenced by historical freeze scripts. Private
  conversational transcripts are not distributed and must not be invented.
- Suitable GPU hardware and an explicit resource/execution policy.

The source inventory lists unresolved imports rather than claiming they are
installed. The CPU requirements are not a model-inference environment lockfile.
Do not use the historical launch scripts as a one-click benchmark runner.

## Preserved scientific boundaries

The release is not evidence of a new successful reproduction. Old restore-risk
closed-loop results must not be presented as verified restore_v2 outcomes.
No missing raw arrays or video are fabricated from confidence intervals or
summary statistics. Source grouping, folds and weights must be inherited from
the relevant original frozen registry, not inferred from the plotting tables.
