# Compatibility policy

The user's goal is to align Core ML as completely as possible with official Laya.
Official behavior, defaults, validation, return structures and tests are the
source of truth. Do not redesign host semantics based on preference or mirror a
behaviorally divergent downstream implementation.

Current baseline: NandhaKishorM/laya v0.4.1,
1adc59f7e371deb601fcfa18a14e25db238addcc. CI checks it out into .upstream.

Keep host changes traceable to upstream. Limit adaptations to the Core ML tensor,
artifact, device and resource boundaries; document every material incompatibility
in docs/MIGRATION_0_4.md. Preserve explicit failures for unsupported backends.
Use upstream assertions and differential tests to distinguish upstream behavior
from a regression. Do not change an expected value merely to make a test pass.

Before releasing, run lint, portable tests, macOS conversion/inference tests,
build/Twine validation, a clean inference-only wheel smoke test, and the real
model fixture/stability checks. Release only after CI passes. Never put credentials
in files, commits, logs or release notes.
