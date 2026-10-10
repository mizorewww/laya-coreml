"""Strict shape handling for the exported Core ML signature."""

import numpy as np


def collate_items(items, pad_id, *, shape, pad_to_multiple=16, max_length=None):
    if not items or len(items) > shape["batch_size"]:
        raise ValueError("Batch must contain 1..exported batch_size questions")
    length = max(len(item["ids"]) for item in items)
    limit = shape["max_length"]
    if length > limit:
        raise ValueError(f"Input has {length} tokens, but this export supports at most {limit}")
    if any(len(item["markers"]) > shape["max_options"] for item in items):
        raise ValueError("Question exceeds the exported max_options; convert with a larger value")
    if shape.get("lengths"):
        length = next(n for n in shape["lengths"] if n >= length)
    elif shape["flexible"]:
        length = min(limit, max(shape["min_length"], ((length + 15) // 16) * 16))
    else:
        length = limit
    b, k = shape["batch_size"], shape["max_options"]
    arrays = {
        "input_ids": np.full((b, length), pad_id, np.int32),
        "attention_mask": np.zeros((b, length), np.int32),
        "marker_pos": np.zeros((b, k), np.int32),
        "marker_mask": np.zeros((b, k), np.int32),
        "qtype": np.zeros((b,), np.int32),
    }
    parallel = "layout" in items[0]
    if any(("layout" in item) != parallel for item in items):
        raise ValueError("Cannot mix sequential and parallel items")
    if parallel:
        arrays["position_ids"] = np.zeros((b, length), np.int32)
        arrays["option_ids"] = np.zeros((b, length), np.int32)
    # A partially occupied fixed batch still needs one valid key in each dummy row.
    arrays["attention_mask"][:, 0] = 1
    for row, item in enumerate(items):
        n, count = len(item["ids"]), len(item["markers"])
        arrays["input_ids"][row, :n] = item["ids"]
        arrays["attention_mask"][row, :n] = 1
        arrays["marker_pos"][row, :count] = item["markers"]
        arrays["marker_mask"][row, :count] = 1
        arrays["qtype"][row] = item["qtype"]
        if parallel:
            for name in ("position_ids", "option_ids"):
                if len(item["layout"][name]) != n:
                    raise ValueError("Layout length must match token length")
                arrays[name][row, :n] = item["layout"][name]
    return arrays
