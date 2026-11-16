"""The native checkpoint temperature reader clamps like upstream and keeps the raw values.

A fitted temperature below 1 sharpens logits. The shipped `choice:11+` bucket is 0.1006,
which turned a 0.24 top probability into 0.99 confidence on 13-option skill routing.
"""

import pytest

from laya_coreml.common import TEMP_MIN, clamp_temperature, read_temperatures


def test_read_temperatures_clamps_and_keeps_raw():
    cfg = {
        "temperature": [1.3, 1.1, 2.0],
        "temperature_by_options": {"choice:2": 1.7, "choice:11+": 0.10058280825614929},
    }
    with pytest.warns(RuntimeWarning, match="clamping choice:11+"):
        temperature, by_options, raw, raw_by_options = read_temperatures(cfg)
    assert by_options["choice:11+"] == TEMP_MIN
    assert raw_by_options["choice:11+"] == 0.10058280825614929
    assert by_options["choice:2"] == 1.7  # legitimate value untouched
    assert temperature == [1.3, 1.1, 2.0]
    assert raw == [1.3, 1.1, 2.0]


def test_read_temperatures_defaults_and_no_warning_when_clean():
    temperature, by_options, raw, raw_by_options = read_temperatures({})
    assert temperature == [1.0, 1.0, 1.0]
    assert by_options == {}
    assert raw == [1.0, 1.0, 1.0]
    assert raw_by_options == {}


@pytest.mark.parametrize(
    "cfg",
    [
        {"temperature": [1.0, float("nan"), 1.0]},
        {"temperature": [1.0, 0.0, 1.0]},  # non-positive
        {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {"choice:2": -1.0}},
    ],
)
def test_read_temperatures_matches_official_clamping(cfg):
    import warnings

    with warnings.catch_warnings(record=True):
        temperature, buckets, raw, raw_buckets = read_temperatures(cfg)
    assert temperature == [clamp_temperature(t) for t in cfg["temperature"]]
    assert buckets == {
        k: clamp_temperature(v) for k, v in cfg.get("temperature_by_options", {}).items()
    }
    assert raw is cfg["temperature"]


def test_temperature_shape_matches_official():
    with pytest.raises(ValueError, match="list of 3"):
        read_temperatures({"temperature": [1.0, 1.0]})


@pytest.mark.parametrize("value", [10**400, -(10**400), 2**1024])
def test_temperature_wider_than_a_double_loads_with_neutral_fallback(value):
    cfg = {"temperature": [value, 1.1, 2.0], "temperature_by_options": {"choice:2": value}}
    with pytest.warns(RuntimeWarning, match=str(value)):
        temperature, by_options, raw, raw_by_options = read_temperatures(cfg)
    assert temperature == [1.0, 1.1, 2.0]
    assert by_options == {"choice:2": 1.0}
    assert raw[0] == value and raw_by_options["choice:2"] == value
