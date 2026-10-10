import json

import pytest
import torch
from safetensors.torch import save_file

from laya_coreml.torch_model import DecisionModel


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(2026)
    cfg = {
        "model_type": "modernbert",
        "hidden_size": 32,
        "intermediate_size": 48,
        "vocab_size": 100,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "local_attention": 8,
        "global_attn_every_n_layers": 3,
    }
    agent = {
        "head_layers": 2,
        "act_costs": {"escalate": 0.5},
        "max_len": 64,
        "head_max_len": 32,
        "encoder": "tiny-test",
        "temperature": [1.0, 1.0, 1.0],
    }
    model = DecisionModel(cfg, agent, 64).eval()
    # The original checkpoint initializes in_proj_weight. This tiny fixture must too.
    for name, parameter in model.named_parameters():
        if "in_proj" in name:
            torch.nn.init.normal_(parameter, std=0.03)
    (tmp_path / "encoder").mkdir()
    (tmp_path / "encoder/config.json").write_text(json.dumps(cfg))
    (tmp_path / "rl_agent_config.json").write_text(json.dumps(agent))
    (tmp_path / "tokenizer").mkdir()
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    tok = Tokenizer(
        WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[PAD]": 4}, unk_token="[UNK]")
    )
    tok.save(str(tmp_path / "tokenizer/tokenizer.json"))
    (tmp_path / "tokenizer/tokenizer_config.json").write_text(
        json.dumps(
            {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
                "pad_token": "[PAD]",
            }
        )
    )
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    return tmp_path, model
