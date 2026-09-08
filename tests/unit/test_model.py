"""Unit tests for the auditable analytical formulas."""

from math import isclose

import pytest

from kvcache_offload_deploy_estimate.model import (
    PP2_TP8,
    PP8_TP2,
    average_burst_ttft,
    pipeline_efficiency,
    pp8_to_pp2_throughput_ratio,
    prefill_pp8_to_pp2_ratio,
    selected_entries,
    selected_payload_bytes_per_gpu,
)


def test_pp8_tp2_and_pp2_tp8_have_same_per_gpu_layer_share() -> None:
    assert PP8_TP2.total_gpus == PP2_TP8.total_gpus == 16
    assert PP8_TP2.average_layer_share_per_gpu == PP2_TP8.average_layer_share_per_gpu
    assert PP8_TP2.average_layer_share_per_node == 0.5


def test_pipeline_efficiency_matches_forward_wave_formula() -> None:
    assert pipeline_efficiency(8, 1) == 1 / 8
    assert pipeline_efficiency(8, 28) == 28 / 35


def test_selected_entries_match_128k_and_256k_cases() -> None:
    assert selected_entries(131_072) == 2_048
    assert selected_entries(262_144) == 4_096


def test_selected_payload_is_replicated_across_pure_tp_ranks() -> None:
    pp8 = selected_payload_bytes_per_gpu(
        context_tokens=131_072,
        cache_layers=61,
        cached_width=576,
        bytes_per_value=2.0,
        topology=PP8_TP2,
    )
    pp2 = selected_payload_bytes_per_gpu(
        context_tokens=131_072,
        cache_layers=61,
        cached_width=576,
        bytes_per_value=2.0,
        topology=PP2_TP8,
    )
    assert pp2 == 4 * pp8
    assert isclose(pp8 / (1024 * 1024), 17.15625)


@pytest.mark.parametrize(
    ("microbatches", "expected"),
    [(1, 0.24), (14, 1.03 * 15 / 21), (28, 1.03 * 29 / 35)],
)
def test_pp8_to_pp2_ratio(microbatches: int, expected: float) -> None:
    assert isclose(pp8_to_pp2_throughput_ratio(microbatches), expected)


def test_prefill_ratio_and_burst_ttft() -> None:
    assert isclose(prefill_pp8_to_pp2_ratio(32), 1.03 * 33 / 39)
    assert average_burst_ttft(16.6, 14) == pytest.approx(124.5)


def test_invalid_pipeline_inputs_fail() -> None:
    with pytest.raises(ValueError):
        pipeline_efficiency(8, 0)
