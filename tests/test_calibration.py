"""Calibration temperature clamping, ported from upstream v0.3.5 (#35/#42).

A fitted temperature below 1 sharpens logits. The shipped `choice:11+` bucket is 0.1006,
which turned a 0.24 top probability into 0.99 confidence on 13-option skill routing.
"""

import pytest

from laya_coreml.common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    clamp_temperature,
    read_temperatures,
    temp_bucket,
)


@pytest.mark.parametrize(
    "value,want",
    [
        (0.1006, 0.5),  # pathological sharpening
        (0.10058280825614929, TEMP_MIN),  # the shipped choice:11+ bucket
        (1.7601518630981445, 1.7601518630981445),  # legitimate value untouched
        (1.0, 1.0),
        (9.0, TEMP_MAX),
        (0.0, TEMP_MIN),
        (-3.0, TEMP_MIN),
        (None, 1.0),
        (False, 1.0),
        (True, 1.0),
        ("x", 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
    ],
)
def test_clamp_temperature(value, want):
    assert clamp_temperature(value) == want


def test_clamp_bounds_are_sane_and_bucket_matches_reported_case():
    assert TEMP_MIN <= 1.0 <= TEMP_MAX
    # 13 options is the bucket the reported skill-router landed in
    assert temp_bucket(QTYPES["choice"], 13) == "choice:11+"


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
