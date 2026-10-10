import sys

import numpy as np
import pytest
import torch

from laya_coreml.inputs import collate_items


def batch(length=31, batch_size=3, lengths=None):
    items = [
        {"ids": list(range(length)), "markers": [3, 8, 10][: i + 1], "qtype": i}
        for i in range(batch_size)
    ]
    shape = {
        "batch_size": 3,
        "max_length": 64,
        "min_length": 16,
        "max_options": 4,
        "flexible": True,
        "lengths": lengths,
    }
    return {k: torch.from_numpy(v) for k, v in collate_items(items, 4, shape=shape).items()}


def test_traced_graph_handles_new_lengths_types_and_markers(checkpoint):
    _, model = checkpoint
    with torch.inference_mode():
        traced = torch.jit.trace(model, tuple(batch().values()))
        for length in (17, 47, 64):
            inputs = batch(length)
            actual, expected = traced(*inputs.values()), model(**inputs)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e)


def test_padding_values_cannot_affect_valid_outputs(checkpoint):
    _, model = checkpoint
    inputs = batch(17)
    with torch.inference_mode():
        expected = model(**inputs)
        inputs["input_ids"][:, 17:] = 99
        actual = model(**inputs)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(sys.platform != "darwin", reason="Core ML runtime requires macOS")
def test_real_coreml_conversion_and_dynamic_padding(checkpoint):
    from laya_coreml import Agent
    from laya_coreml.convert import convert

    source, model = checkpoint
    output = source / "export"
    convert(source, output, batch_size=3, max_options=4)
    agent = Agent(output, compute_units="cpu_gpu")
    for length in (17, 47, 64):
        inputs = batch(length, lengths=agent.shape["lengths"])
        with torch.inference_mode():
            expected = model(**inputs)
        actual = agent.forward({k: v.numpy() for k, v in inputs.items()})
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e.numpy(), atol=0.015, rtol=0.015)
    result = agent.predict(
        "sample", {"q": {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}}
    )
    assert result["usage"]["output_tokens"] == 0
    assert set(result["answers"]["q"]["probabilities"]) == {"a", "b"}
    with pytest.raises(FileExistsError):
        convert(source, output)
