# Compatibility policy

Align Core ML as completely as possible with official Laya. Official behavior,
defaults, validation, return structures and tests are the source of truth. Do not
redesign host semantics based on preference.

Current baseline: NandhaKishorM/laya v0.4.17,
fe9a6c35023a7d94102e7feaaeaae8857d9d1b07. CI checks it out into .upstream.

Keep host changes traceable to upstream. Limit adaptations to the Core ML tensor,
artifact, device and resource boundaries. Preserve explicit failures for unsupported
backends. Use upstream assertions and differential tests to distinguish upstream
behavior from a regression. Do not change an expected value merely to make a test pass.

## Fixing upstream defects

A confirmed bug that upstream still has may be fixed here, including by porting the
same fix from the sibling laya-mlx project, when all of the following hold:

- The defect is reproduced against the pinned upstream code, not inferred.
- The fix is minimal and keeps upstream's public API, defaults and return structures.
- Regression tests fail before the fix and pass after it; existing upstream assertions
  stay unchanged.
- The divergence is listed in the "Deliberate divergences from upstream" table of the
  current migration document (docs/MIGRATION_0_4_17.md), with its tests.

When a later upstream release fixes the same defect, adopt upstream's version and
remove the table entry.

## Documentation

Record every material incompatibility or divergence in the current migration
document; docs/MIGRATION_0_4.md holds those carried forward from earlier migrations.

## Releasing

Before releasing, run lint, portable tests, macOS conversion/inference tests,
build/Twine validation, a clean inference-only wheel smoke test, and the real
model fixture/stability checks. Release only after CI passes. Never put credentials
in files, commits, logs or release notes.
