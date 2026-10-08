# Release preparation audit

Prepared on 2026-10-08. This audit concerns a new source-release copy only.
Original experiments, protocols, weights, arrays, and reports were not modified.

## Completed checks

- 80 historical method/dependency Python files exported from an explicit allowlist
  and statically resolved local imports; no ambiguous local imports remain in the
  export inventory. External modules and assets remain separately required.
- Five existing plotting/helper Python files exported, plus nine saved CSV inputs.
- All 85 exported Python files retain identical ASTs after normalizing only the
  documented path substitutions, relocation-helper import and released-data hash
  guards. The per-file results are in `provenance/algorithm_preservation_check.json`.
- All CSV cells are unchanged except personal absolute-path strings; no summary
  estimate, CI, per-source response, dose, fold, weight, or cohort was regenerated.
- Three packaging/path tests passed. All 90 Python files in this release parsed.
- The physical-response, Joint-attribution/restoration and reduced-computation
  figures rendered successfully from their saved inputs using NumPy 1.26.4,
  pandas 2.2.3 and Matplotlib 3.5.3. Six PDF/PNG outputs are included.
- Public file selection scanned for personal absolute paths, common credential
  patterns, oversized files and accidental runtime artifacts. No such pattern
  was detected. This is not a guarantee against every possible identity linkage.
- The FastWAM MIT license file is byte-identical to its local source notice.

## Not performed / not claimed

- No model import/execution, simulator creation/step, donor generation, rollout,
  scientific fit or bootstrap was performed as a release check.
- Reference model scripts were syntax-checked, not dynamically validated in a
  clean deployment environment. The quick start validates saved-result plotting,
  not a full regeneration of the paper's experiments.
- Checkpoints, raw data, complete frozen manifests, external private instructions
  and third-party deployment stacks are not bundled.
- Git history anonymity is a separate check from file-content anonymization.
  No remote history was rewritten or publication performed by these checks.
- The original audit-code license remains an owner decision; the upstream MIT
  notice must not be interpreted as granting a blanket license to everything.

Generated figure audit files containing local output paths are excluded by the
release `.gitignore`. The final release manifest indexes only the public file
selection, excludes itself, and is generated after those files stop changing.
