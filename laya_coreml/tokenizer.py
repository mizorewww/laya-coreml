"""Load the checkpoint's Rust tokenizer without importing Transformers or torch."""

import json
from pathlib import Path

from tokenizers import Tokenizer as Backend


class Tokenizer:
    def __init__(self, path):
        path = Path(path)
        self.backend = Backend.from_file(str(path / "tokenizer.json"))
        self.backend.no_padding()
        self.backend.no_truncation()
        config = json.loads((path / "tokenizer_config.json").read_text())
        for name in ("cls_token", "sep_token", "pad_token", "mask_token"):
            value = config.get(name)
            if isinstance(value, dict):
                value = value.get("content")
            token_id = self.backend.token_to_id(value) if isinstance(value, str) else None
            if token_id is None:
                raise ValueError(f"Tokenizer is missing a valid {name}")
            setattr(self, name, value)
            setattr(self, name + "_id", token_id)

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = self.backend.encode(text, add_special_tokens=add_special_tokens).ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return self.backend.decode(list(ids), skip_special_tokens=skip_special_tokens, **kwargs)
