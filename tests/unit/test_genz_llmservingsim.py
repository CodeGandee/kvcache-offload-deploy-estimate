"""Tests for the GenZ-to-LLMServingSim model-core profile bridge."""

import csv
from pathlib import Path

import pytest

from kvcache_offload_deploy_estimate.genz_llmservingsim import (
    MODEL_SPECS,
    analytical_decode_ms,
    analytical_decode_stage_ms,
    expected_distinct_experts,
    generate_profile_bundles,
    llmservingsim_decode_ms,
)


def test_official_configs_define_all_four_targets() -> None:
    assert set(MODEL_SPECS) == {"kimi", "glm", "glm-flash", "deepseek-flash"}
    assert MODEL_SPECS["glm"].layers == 78
    assert MODEL_SPECS["glm-flash"].layers == 45
    assert MODEL_SPECS["glm-flash"].linear_attention_layers == 34
    assert MODEL_SPECS["glm-flash"].cache_layers == 11


def test_expected_moe_occupancy_is_bounded_and_increases() -> None:
    spec = MODEL_SPECS["glm"]
    values = [expected_distinct_experts(spec, sequences) for sequences in (1, 8, 64)]
    assert values == sorted(values)
    assert 0 < values[0] < values[-1] < spec.experts


@pytest.mark.parametrize("model_key", sorted(MODEL_SPECS))
def test_pipeline_stage_times_sum_to_full_model(model_key: str) -> None:
    assert sum(analytical_decode_stage_ms(model_key, 8)) == pytest.approx(
        analytical_decode_ms(model_key, 8)
    )


@pytest.mark.parametrize("model_key", sorted(MODEL_SPECS))
def test_llmservingsim_exact_row_matches_generated_genz_time(model_key: str) -> None:
    generate_profile_bundles()
    expected_ms = analytical_decode_ms(model_key, 8)
    assert llmservingsim_decode_ms(model_key, 8) == pytest.approx(expected_ms, abs=5e-5)


def test_generated_bundle_uses_llmservingsim_per_sequence_schema() -> None:
    paths = generate_profile_bundles()
    profile = next(path for path in paths if path.name == "per_sequence.csv")
    with Path(profile).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0].keys() == {"layer", "sequences", "time_us"}
    assert rows[0]["layer"] == "model_core"
    assert rows[0]["sequences"] == "1"
    assert rows[-1]["sequences"] == "4096"
