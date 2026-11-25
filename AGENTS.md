# Compatibility policy

Align Core ML as completely as possible with official Laya. Official behavior,
defaults, validation, return structures and tests are the source of truth. Do not
redesign host semantics based on preference.

Current baseline: NandhaKishorM/laya v0.5.4,
60a6ce910b4a2edb2b77e00d8fddeb24349610c8. CI checks it out into .upstream.

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
- The divergence is listed in the "Deliberate divergences" table of the compatibility
  record (docs/UPSTREAM.md), with its tests.

When a later upstream release fixes the same defect, adopt upstream's version and
remove the table entry.

## Documentation

Record every material incompatibility or divergence in docs/UPSTREAM.md. It is one
evergreen record: update it in place on each sync instead of adding a per-version file.

## Releasing

Before releasing, run lint, portable tests, macOS conversion/inference tests,
build/Twine validation, a clean inference-only wheel smoke test, and the real
model fixture/stability checks. Release only after CI passes. Never put credentials
in files, commits, logs or release notes.

Upstream ports are the exception: .github/upstream-port.md releases them
automatically once CI passes, and the wheel smoke test and real-model checks run
afterwards on Apple hardware.
