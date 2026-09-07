"""Tests for the LLMServingSim-compatible ShadowKV event model."""

import pytest

from kvcache_offload_deploy_estimate.llmservingsim_shadowkv import (
    SCENARIOS,
    OraclePrefetch,
    estimate_point,
    stage_layer_counts,
    stored_bytes_per_value,
)


def test_oracle_separates_recall_from_precision() -> None:
    prefetch, jit = OraclePrefetch(recall=0.8, precision=0.8).traffic_factors(reuse=0.0)
    assert prefetch == pytest.approx(1.0)
    assert jit == pytest.approx(0.2)
    assert prefetch + jit == pytest.approx(1.2)


def test_oracle_deduplicates_temporal_reuse() -> None:
    prefetch, jit = OraclePrefetch(recall=0.8, precision=0.8).traffic_factors(reuse=0.6)
    assert prefetch == pytest.approx(0.4)
    assert jit == pytest.approx(0.08)
    assert OraclePrefetch(recall=1.0, precision=1.0).traffic_factors(reuse=0.6) == (0.4, 0.0)


def test_stage_counts_come_from_upstream_llmservingsim() -> None:
    assert stage_layer_counts(61, 8) == (7, 7, 8, 8, 8, 8, 8, 7)
    assert stage_layer_counts(78, 8) == (9, 10, 10, 10, 10, 10, 10, 9)


def test_fp8_storage_includes_scales() -> None:
    assert stored_bytes_per_value("fp8") == pytest.approx(1.03125)
    assert stored_bytes_per_value("bf16") == 2.0


def test_prefetch_beats_fetch_at_decode_at_full_load() -> None:
    scenario = SCENARIOS[0]
    ahead = estimate_point(
        scenario,
        load_index=4,
        policy="oracle-prefetch",
        storage="fp8",
        sensitivity_samples=0,
    )
    fetch = estimate_point(
        scenario,
        load_index=4,
        policy="fetch-at-decode",
        storage="fp8",
        sensitivity_samples=0,
    )
    assert ahead.tpot_ms < fetch.tpot_ms
    assert ahead.aggregate_tps > fetch.aggregate_tps


def test_prefetch_trace_contains_background_and_critical_events() -> None:
    point = estimate_point(
        SCENARIOS[2],
        load_index=2,
        policy="oracle-prefetch",
        storage="fp8",
        sensitivity_samples=0,
    )
    assert any(
        event.name == "oracle_prefetch" and event.background_ms > 0 for event in point.trace_events
    )
    assert any(event.critical_ms > 0 for event in point.trace_events)


def test_authoritative_perfect_oracle_is_an_explicit_upper_bound() -> None:
    scenario = SCENARIOS[0]
    advisory = estimate_point(
        scenario,
        load_index=4,
        policy="oracle-prefetch",
        storage="fp8",
        sensitivity_samples=0,
    )
    authoritative = estimate_point(
        scenario,
        load_index=4,
        policy="oracle-prefetch",
        storage="fp8",
        oracle=OraclePrefetch(recall=1.0, precision=1.0, verify_with_landmarks=False),
        sensitivity_samples=0,
    )
    assert authoritative.tpot_ms < advisory.tpot_ms


def test_fetch_at_decode_never_skips_landmark_selection() -> None:
    scenario = SCENARIOS[0]
    trusted = estimate_point(
        scenario,
        load_index=4,
        policy="fetch-at-decode",
        storage="fp8",
        oracle=OraclePrefetch(recall=1.0, precision=1.0, verify_with_landmarks=False),
        sensitivity_samples=0,
    )
    ordinary = estimate_point(
        scenario,
        load_index=4,
        policy="fetch-at-decode",
        storage="fp8",
        sensitivity_samples=0,
    )
    assert trusted.tpot_ms == ordinary.tpot_ms
