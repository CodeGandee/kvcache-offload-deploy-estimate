"""Tests for the LLMServingSim-compatible ShadowKV event model."""

import math
from itertools import pairwise

import pytest

from kvcache_offload_deploy_estimate.llmservingsim_shadowkv import (
    SCENARIOS,
    A800Host,
    OraclePrefetch,
    PolicyName,
    StorageName,
    _optimal_microbatch_size,
    _transfer_time_ms,
    estimate_no_shadowkv_point,
    estimate_point,
    estimate_residency_scan,
    native_cache_gib_per_gpu,
    no_shadowkv_max_users,
    residency_scan,
    stage_layer_counts,
    stored_bytes_per_value,
)


def test_measured_tp2_pair_bandwidth_limits_concurrent_h2d() -> None:
    hardware = A800Host()
    elapsed = _transfer_time_ms(
        stage_bytes_per_gpu=1e9,
        factor=1.0,
        users=1,
        tp_size=2,
        hardware=hardware,
    )
    assert elapsed == pytest.approx(2e12 / (25.4e9))


def test_tp2_pair_measurement_is_not_assumed_for_tp4() -> None:
    hardware = A800Host()
    elapsed = _transfer_time_ms(
        stage_bytes_per_gpu=1e9,
        factor=1.0,
        users=1,
        tp_size=4,
        hardware=hardware,
    )
    assert elapsed == pytest.approx(1e12 / (22.0e9))


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


def test_72k_context_is_part_of_each_model_family() -> None:
    assert {scenario.id for scenario in SCENARIOS if scenario.context_tokens == 73_728} == {
        "kimi-72",
        "glm-72",
        "glm-flash-72",
        "flash-72",
    }


def test_residency_scan_reports_whole_layer_ratios() -> None:
    kimi_ten = residency_scan(61)[1]
    glm_ten = residency_scan(78)[1]
    flash_ten = residency_scan(21)[1]
    assert kimi_ten == pytest.approx((0.1, 6, 6 / 61))
    assert glm_ten == pytest.approx((0.1, 8, 8 / 78))
    assert flash_ten == pytest.approx((0.1, 2, 2 / 21))
    assert residency_scan(61)[-1] == (1.0, 61, 1.0)


def test_full_residency_removes_host_payload_and_reduces_tpot() -> None:
    scenario = next(item for item in SCENARIOS if item.id == "glm-128")
    points = estimate_residency_scan(
        scenario,
        users=scenario.max_users,
        policy="fetch-at-decode",
        storage="fp8",
    )
    assert points[-1].estimate.tpot_ms < points[0].estimate.tpot_ms
    assert sum(event.bytes_moved for event in points[-1].estimate.trace_events) == 0


def test_mtp_requires_checkpoint_component_and_two_accepts_beat_one() -> None:
    with pytest.raises(ValueError, match="no checkpoint MTP component"):
        estimate_point(
            next(item for item in SCENARIOS if item.id == "kimi-128"),
            load_index=4,
            policy="oracle-prefetch",
            storage="fp8",
            sensitivity_samples=0,
            mtp_accepted_tokens=1,
        )
    glm = next(item for item in SCENARIOS if item.id == "glm-128")
    one = estimate_point(
        glm,
        load_index=4,
        policy="oracle-prefetch",
        storage="fp8",
        sensitivity_samples=0,
        mtp_accepted_tokens=1,
    )
    two = estimate_point(
        glm,
        load_index=4,
        policy="oracle-prefetch",
        storage="fp8",
        sensitivity_samples=0,
        mtp_accepted_tokens=2,
    )
    assert two.tpot_ms < one.tpot_ms


@pytest.mark.parametrize(
    "scenario_id", ["kimi-72", "kimi-128", "kimi-256", "glm-72", "glm-128", "glm-256"]
)
@pytest.mark.parametrize("policy", ["oracle-prefetch", "fetch-at-decode"])
@pytest.mark.parametrize("storage", ["bf16", "fp8"])
def test_pp8_per_user_throughput_does_not_improve_with_load(
    scenario_id: str,
    policy: PolicyName,
    storage: StorageName,
) -> None:
    scenario = next(item for item in SCENARIOS if item.id == scenario_id)
    points = [
        estimate_point(
            scenario,
            load_index=index,
            policy=policy,
            storage=storage,
            sensitivity_samples=0,
        )
        for index in range(5)
    ]
    assert all(right.per_user_tps <= left.per_user_tps for left, right in pairwise(points))


def test_generated_moe_profile_grows_with_distinct_experts() -> None:
    scenario = next(item for item in SCENARIOS if item.id == "glm-128")
    assert scenario.reference_tpot_for_users(20) > scenario.reference_tpot_for_users(1)
    assert scenario.reference_tpot_for_users(40) > scenario.reference_tpot_for_users(20)

    scans = [
        estimate_residency_scan(
            scenario,
            users=users,
            policy="oracle-prefetch",
            storage="fp8",
        )
        for users in (1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40)
    ]
    for residency_index in range(len(scans[0])):
        per_user = [scan[residency_index].estimate.per_user_tps for scan in scans]
        assert all(right <= left for left, right in pairwise(per_user))


def test_pp8_scheduler_uses_enough_groups_to_fill_pipeline() -> None:
    scenario = next(item for item in SCENARIOS if item.id == "glm-128")
    microbatch = _optimal_microbatch_size(
        scenario,
        users=scenario.max_users,
        policy="oracle-prefetch",
        storage="fp8",
        oracle=OraclePrefetch(),
        reuse=0.60,
        resident_layers=0,
        mtp_accepted_tokens=0,
    )
    assert microbatch <= math.ceil(scenario.max_users / scenario.pp_size)


def test_no_shadowkv_admission_is_a_full_native_cache_hbm_limit() -> None:
    scenario = next(item for item in SCENARIOS if item.id == "glm-256")
    bf16_max = no_shadowkv_max_users(scenario, storage="bf16")
    fp8_max = no_shadowkv_max_users(scenario, storage="fp8")
    assert bf16_max == 14
    assert fp8_max == 28
    assert native_cache_gib_per_gpu(scenario, storage="bf16", users=bf16_max) <= 22.1
    assert native_cache_gib_per_gpu(scenario, storage="bf16", users=bf16_max + 1) > 22.0


def test_no_shadowkv_trace_has_no_host_transfer_or_landmark_selection() -> None:
    scenario = next(item for item in SCENARIOS if item.id == "kimi-128")
    point = estimate_no_shadowkv_point(
        scenario,
        users=4,
        storage="fp8",
        sensitivity_samples=0,
    )
    assert point.tpot_ms > 0
    assert point.aggregate_tps == pytest.approx(point.users * point.per_user_tps)
    assert {event.name for event in point.trace_events} == {"native_attention_hbm"}


def test_no_shadowkv_rejects_requests_past_oom_boundary() -> None:
    scenario = next(item for item in SCENARIOS if item.id == "flash-256")
    maximum = no_shadowkv_max_users(scenario, storage="bf16")
    with pytest.raises(ValueError, match="HBM admission ceiling"):
        estimate_no_shadowkv_point(
            scenario,
            users=maximum + 1,
            storage="bf16",
            sensitivity_samples=0,
        )
