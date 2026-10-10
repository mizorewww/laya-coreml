"""Resolve a local bundle or a Hugging Face snapshot before inference starts."""

import os
from pathlib import Path

DEFAULT_MODEL = "convaiinnovations/laya"

# Audited against official v0.4.1: source weights and rl_agent_config.json are identical.
BUILTIN_BUNDLES = {
    ("convaiinnovations/laya", None): "aac6fef/laya-coreml",
    ("convaiinnovations/laya", "multilingual"): "aac6fef/laya-multilingual-coreml",
    ("convaiinnovations/laya", "typed-decisions"): "aac6fef/laya-typed-decisions-coreml",
    ("convaiinnovations/laya-multilingual", None): "aac6fef/laya-multilingual-coreml",
    ("convaiinnovations/laya-typed-decisions", None): "aac6fef/laya-typed-decisions-coreml",
}
BUNDLE_REVISIONS = {
    "aac6fef/laya-coreml": "fff78b2d9750c6b748fe8c90fcbf8bed0a1522a9",
    "aac6fef/laya-multilingual-coreml": "8139e9089273319512c730218903784074133187",
    "aac6fef/laya-typed-decisions-coreml": "28d24fa8d67a3264556b23391ec6c3fd98573056",
}
SOURCE_REVISIONS = {
    "convaiinnovations/laya": {"7b928d828b7b0e022f929d9bd2e44165aa270148"},
    "convaiinnovations/laya-multilingual": {"1720e3e3357cfe1e281542e223f8273b0890ca34"},
    "convaiinnovations/laya-typed-decisions": {"e929ae5cf69bc34259cd2f95c9e91145b818b1f0"},
}


def resolve_checkpoint(model, *, revision=None, local_files_only=False, token=None, subfolder=None):
    path = Path(model).expanduser()
    if path.is_dir():
        return path / subfolder if subfolder else path
    if isinstance(model, Path) or path.is_absolute() or str(model).startswith((".", "~")):
        raise FileNotFoundError(f"Local model directory does not exist: {model}")
    from .revisions import PINNED_REVISIONS, resolve_revision

    key = (str(model).lower(), subfolder or None)
    if (revision or os.environ.get("LAYA_REVISION", "")).strip() == "reviewed" and str(
        model
    ).lower() in BUNDLE_REVISIONS:
        requested = BUNDLE_REVISIONS[str(model).lower()]
    else:
        requested = resolve_revision(str(model), revision)
    if key in BUILTIN_BUNDLES:
        target = BUILTIN_BUNDLES[key]
        # Source and converted repositories have different commit histories. Only
        # reviewed source revisions with identical weights/config can be translated.
        if requested is not None:
            source_repo = str(model).lower()
            accepted = {PINNED_REVISIONS.get(source_repo), *SOURCE_REVISIONS[source_repo]}
            if requested not in accepted:
                raise ValueError(
                    "No reviewed Core ML conversion for source revision %r; use an explicit "
                    "converted repository and its revision, or convert that source checkpoint."
                    % requested
                )
            requested = BUNDLE_REVISIONS[target]
        model, subfolder = target, None
    revision = requested
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            str(model),
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=[
                (subfolder + "/" if subfolder else "") + pattern
                for pattern in [
                    "coreml_config.json",
                    "rl_agent_config.json",
                    "encoder/config.json",
                    "tokenizer/*",
                    "model.mlpackage/**",
                    "host_weights.safetensors",
                ]
            ],
        )
    )

    return path / subfolder if subfolder else path
