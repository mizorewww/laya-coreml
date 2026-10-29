"""Batch grouping must preserve the Python types exposed in public answers."""

from copy import deepcopy

import pytest

from laya_coreml import Router


class LabelAgent:
    def __init__(self):
        self.batches = []

    def system_one(self, state, questions, **kwargs):
        answers = {}
        for qid, question in questions.items():
            labels = list(question["criteria"])
            answers[qid] = {
                "type": "choice",
                "choice": labels[0],
                "probabilities": {label: float(i == 0) for i, label in enumerate(labels)},
            }
        return {"answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}

    def predict_batch(self, states, questions, **kwargs):
        self.batches.append((list(states), deepcopy(questions)))
        return [self.system_one(state, questions) for state in states]


def question(qid="q", label="selected"):
    return {
        qid: {
            "type": "choice",
            "instructions": "Choose",
            "criteria": {label: "Select this", "other": "Do not select this"},
        }
    }


def typed_answers(result):
    return [
        (
            type(qid),
            qid,
            type(answer["choice"]),
            answer["choice"],
            [(type(label), label, prob) for label, prob in answer["probabilities"].items()],
        )
        for qid, answer in result["answers"].items()
    ]


@pytest.mark.parametrize("keys", [(1, "1"), (True, "true"), (False, "false")])
@pytest.mark.parametrize("location", ["qid", "label"])
def test_batch_keeps_scalar_key_types(keys, location):
    router = Router()
    agent = LabelAgent()
    router.attach("english", agent)
    requests = [
        {"state": str(i), "questions": question(**{location: key}), "model": "english"}
        for i, key in enumerate(keys)
    ]
    expected = [typed_answers(router.predict(**item)) for item in requests]

    actual = router.predict_batch(requests)

    assert [typed_answers(result) for result in actual] == expected
    assert len(agent.batches) == 2


def test_batch_keeps_null_choice_key_distinct_from_string():
    router = Router()
    agent = LabelAgent()
    router.attach("english", agent)
    requests = [
        {"state": "s", "questions": question(label=key), "model": "english"}
        for key in (None, "null")
    ]
    expected = [typed_answers(router.predict(**item)) for item in requests]
    assert [typed_answers(result) for result in router.predict_batch(requests)] == expected
    assert len(agent.batches) == 2


def test_equal_schemas_still_batch_and_option_order_stays_significant():
    router = Router()
    agent = LabelAgent()
    router.attach("english", agent)
    original = question(label=1)
    reordered = deepcopy(original)
    reordered["q"]["criteria"] = dict(reversed(list(original["q"]["criteria"].items())))
    requests = [
        {"state": str(i), "questions": schema, "model": "english"}
        for i, schema in enumerate((original, deepcopy(original), reordered))
    ]
    expected = [typed_answers(router.predict(**item)) for item in requests]
    assert [typed_answers(result) for result in router.predict_batch(requests)] == expected
    assert [states for states, _ in agent.batches] == [["0", "1"], ["2"]]


def test_nested_instruction_key_types_do_not_share_a_batch():
    router = Router()
    agent = LabelAgent()
    router.attach("english", agent)
    requests = []
    for key in (1, "1"):
        schema = question()
        schema["q"]["instructions"] = [{key: "Choose"}]
        requests.append({"state": "s", "questions": schema, "model": "english"})
    router.predict_batch(requests)
    assert len(agent.batches) == 2
