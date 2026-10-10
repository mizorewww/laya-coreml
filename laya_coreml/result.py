"""Laya output semantics, adapted from laya-mlx; see NOTICE."""

import numpy as np

from .common import answer_confidence, collapsed_options, confidence_from_probs, temp_bucket
from .confidence import apply_confidence_gate, check_min_confidence
from .inputs import collate_items


class ResultMixin:
    def system_one(self, state, questions, *, min_confidence=None):
        if min_confidence is not None:
            min_confidence = check_min_confidence(min_confidence)
        items, internal = self.prepare(state, questions)
        answers = {}
        question_ids = list(questions)
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            batch = collate_items(
                chunk,
                self.tok.pad_token_id,
                pad_to_multiple=self.pad_to_multiple,
                max_length=self.cfg.get("max_len", 512),
                shape=self.shape,
            )
            logits, act = self.forward(batch)
            logits, act = np.asarray(logits), np.asarray(act)
            if not np.isfinite(logits).all() or not np.isfinite(act).all():
                raise FloatingPointError("Non-finite Core ML outputs")
            act = np.exp(act - act.max(axis=-1, keepdims=True))
            act /= act.sum(axis=-1, keepdims=True)
            for row, item in enumerate(chunk):
                qid, q = question_ids[start + row], internal[start + row]
                k, qt = len(item["markers"]), item["qtype"]
                scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
                z = logits[row, :k] / scale
                p = np.exp(z - z.max())
                p /= p.sum()
                answer = {
                    "type": q["t"],
                    "answer_confidence": round(answer_confidence(p, k), 4),
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "action": {"act_probability": round(float(act[row, 0]), 4)},
                }
                if q["t"] == "choice":
                    labels = list(q["crit"])
                    answer.update(
                        choice=labels[int(p.argmax())],
                        probabilities={label: round(float(v), 4) for label, v in zip(labels, p)},
                    )
                elif q["t"] == "score":
                    answer.update(
                        score=round(float((np.arange(k) * p).sum()), 4),
                        legend={str(i): value for i, value in enumerate(q["crit"])},
                        probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    )
                else:
                    answer.update(
                        noul=round(float(p[1]), 4),
                        confidence=round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    )
                answers[qid] = answer
        stats = [item["state_stats"] for item in items]
        dropped = max((s["state_tokens_dropped"] for s in stats), default=0)
        usage = {
            "input_tokens": sum(len(item["ids"]) for item in items),
            "output_tokens": 0,
            "state_tokens": stats[0]["state_tokens"] if stats else 0,
            "state_tokens_dropped": dropped,
            "truncated": dropped > 0,
            "truncated_questions": [qid for qid, s in zip(question_ids, stats) if s["truncated"]],
        }
        collapsed = collapsed_options(question_ids, items)
        if collapsed:
            usage["options"] = collapsed
        result = {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": usage,
        }
        apply_confidence_gate([result], min_confidence)
        return result

    predict = system_one
