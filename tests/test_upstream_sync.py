"""Regression cases for the selective post-v0.3.5 upstream sync."""

import numpy as np
import pytest

from laya_coreml.agent import Agent
from laya_coreml.common import build_sequence, render_options


class WordTokenizer:
    mask_token = "[MASK]"
    mask_token_id, cls_token_id, sep_token_id = 1, 2, 3

    def __init__(self):
        self.vocab = {}
        self.calls = []

    def __call__(self, text, add_special_tokens=False, **kwargs):
        self.calls.append(text)
        return {
            "input_ids": [self.vocab.setdefault(w, len(self.vocab) + 100) for w in text.split()]
        }


def preparation_agent():
    agent = object.__new__(Agent)
    agent.tok = WordTokenizer()
    agent.cfg = {"max_len": 48, "head_max_len": 24}
    return agent


Q = {"q": {"type": "choice", "instructions": "Choose", "criteria": ["yes", "no"]}}


def test_zero_room_left_truncation_keeps_separator():
    tok = WordTokenizer()
    q = Agent._to_internal(Q["q"])
    size = len(build_sequence(tok, "", q, max_len=1000)[0])
    ids, _ = build_sequence(tok, "old newest", q, max_len=size, truncate_left=True)
    assert ids[-1] == tok.sep_token_id
    assert tok.vocab["old"] not in ids
    assert tok.vocab["newest"] not in ids


def test_conversation_keeps_tail_and_string_keeps_head():
    a = preparation_agent()
    state = ["OLD " + "middle " * 100 + "NEWEST"]
    items, _ = a.prepare(state, Q)
    assert a.tok.vocab['NEWEST"]'] in items[0]["ids"]
    assert a.tok.vocab['["OLD'] not in items[0]["ids"]
    assert items[0]["state_stats"]["truncated"]
    items, _ = a.prepare(state[0], Q)
    assert a.tok.vocab["OLD"] in items[0]["ids"]
    assert a.tok.vocab["NEWEST"] not in items[0]["ids"]


def test_noul_labels_unicode_and_named_validation():
    a = preparation_agent()
    q = {
        "type": "noul",
        "instructions": {"text": "这是中文"},
        "criteria": {False: "错误", True: "正确"},
        "labels": {"false": " no ", "true": " yes "},
    }
    items, internal = a.prepare("hello", {"check": q})
    assert render_options(internal[0]) == ["no: 错误", "yes: 正确"]
    assert "这是中文" in internal[0]["ins"]
    assert q["criteria"] == {False: "错误", True: "正确"}
    q["labels"] = {"false": "negative", "true": "positive"}
    assert a.prepare("hello", {"check": q})[0][0]["ids"] != items[0]["ids"]
    for bad in [
        dict(q, criteria={"yes": "wrong"}),
        dict(q, labels={"true": "same", "false": "same"}),
        {"type": [], "instructions": "x"},
        {"type": "score", "instructions": "x", "criteria": [None, "good"]},
        dict(q, instructions=None),
        dict(q, instructions=" "),
    ]:
        with pytest.raises(ValueError, match="question 'check'"):
            a.prepare("state", {"check": bad})
    with pytest.raises(TypeError, match="state"):
        a.prepare(None, Q)


def test_usage_matches_each_question_budget_and_reports_option_collapse():
    a = preparation_agent()
    a.batch_size = 16
    a.shape = {"batch_size": 16, "max_length": 48, "max_options": 8, "flexible": False}
    a.pad_to_multiple = None
    a.tok.pad_token_id = 0
    a.temperature = [1.0] * 3
    a.temperature_by_options = {}
    a.forward = lambda batch: (
        np.zeros(batch["marker_mask"].shape),
        np.zeros((len(batch["qtype"]), 2)),
    )
    qs = {
        "q": Q["q"],
        "many": {
            "type": "choice",
            "instructions": "Choose",
            "criteria": {"shared prefix words " + str(i): None for i in range(6)},
        },
    }
    state = "word " * 100
    out = a.predict(state, qs)
    items, _ = a.prepare(state, qs)
    assert out["usage"]["state_tokens"] == 100
    assert out["usage"]["state_tokens_dropped"] == max(
        i["state_stats"]["state_tokens_dropped"] for i in items
    )
    assert out["usage"]["truncated_questions"] == ["q", "many"]
    assert out["usage"]["options"] == {"many": {"total": 6, "distinct": 1, "tokens_per_option": 4}}
    assert out["answers"]["q"]["answer_confidence"] == 0.5
    assert out["answers"]["q"]["confidence"] == 0.0
    assert a.predict("", {})["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_shared_mixins_and_export_capacity():
    from laya_coreml.ane import ANEAgent
    from laya_coreml.inputs import collate_items
    from laya_coreml.runtime import RuntimeMixin

    for cls in (Agent, ANEAgent):
        assert cls.prepare is RuntimeMixin.prepare
        assert cls.predict is RuntimeMixin.predict
    a = preparation_agent()
    items, _ = a.prepare("word " * 100, Q)
    with pytest.raises(ValueError, match="tokens"):
        collate_items(
            items, 0, shape={"batch_size": 1, "max_length": 32, "max_options": 8, "flexible": False}
        )


def test_state_is_tokenized_once_for_multiple_questions():
    a = preparation_agent()
    a.prepare("unique state", {"first": Q["q"], "second": Q["q"]})
    assert a.tok.calls.count("unique state") == 1
