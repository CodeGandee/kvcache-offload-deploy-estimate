"""ShadowKV trace extension for GenZ-to-LLMServingSim decode studies.

The upstream simulator models ordinary tiered KV blocks.  ShadowKV has a different
per-transformer-block dependency: landmark selection must finish before cache misses
are known, key reconstruction and value fetch overlap, and sparse attention waits for
both.  This module adds those events without modifying the tracked upstream source.

The non-ShadowKV model-forward component comes from official-config operator graphs
evaluated by GenZ and consumed through LLMServingSim's profile-table interface.
Everything added here (selection, prefetch traffic, miss materialization, and FP8 KV
conversion) is calculated explicitly and is therefore independently inspectable.
"""

from __future__ import annotations

import importlib
import math
import random
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

from .genz_llmservingsim import (
    CENTRAL_HARDWARE,
    GENZ_COMMIT,
    LLMSERVINGSIM_COMMIT,
    MODEL_SPECS,
    RooflineHardware,
    analytical_decode_ms,
    analytical_decode_stage_ms,
    analytical_prefill_seconds,
    decode_breakdown,
    llmservingsim_decode_ms,
    llmservingsim_decode_stage_ms,
    profile_manifest,
)

PolicyName = Literal["oracle-prefetch", "fetch-at-decode"]
StorageName = Literal["bf16", "fp8"]


@dataclass(frozen=True, slots=True)
class OraclePrefetch:
    """Prediction quality over entries not already resident from the prior token."""

    recall: float = 0.80
    precision: float = 0.80
    lookahead_tokens: int = 1
    verify_with_landmarks: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.recall <= 1.0:
            raise ValueError("oracle recall must be in [0, 1]")
        if not 0.0 < self.precision <= 1.0:
            raise ValueError("oracle precision must be in (0, 1]")
        if self.lookahead_tokens < 1:
            raise ValueError("lookahead_tokens must be positive")

    def traffic_factors(self, *, reuse: float) -> tuple[float, float]:
        """Return (prefetched, just-in-time) fractions of the selected set.

        The runtime first deducts entries retained from the preceding token.  Recall is
        measured over the remaining required entries; precision adds false positives.
        """

        if not 0.0 <= reuse <= 1.0:
            raise ValueError("reuse must be in [0, 1]")
        missing = 1.0 - reuse
        prefetched = missing * self.recall / self.precision
        just_in_time = missing * (1.0 - self.recall)
        return prefetched, just_in_time


@dataclass(frozen=True, slots=True)
class A800Host:
    """Usable bandwidth assumptions, using the A100 profile as an Ampere proxy."""

    hbm_stream_gbps: float = 1765.0
    decode_gemm_mbu: float = 0.516
    peak_bf16_tflops: float = 312.0
    bf16_compute_efficiency: float = 0.40
    pcie_per_gpu_gbps: float = 22.0
    pcie_tp2_pair_gbps: float = 25.4
    host_dram_gbps: float = 180.0
    fp8_dequant_gbps: float = 455.0
    int4_dequant_gbps: float = 455.0
    nvlink_latency_ms: float = 0.031
    nvlink_payload_gbps: float = 274.0
    ib_latency_ms: float = 0.012
    ib_payload_gbps: float = 40.0


def _roofline_hardware(hardware: A800Host) -> RooflineHardware:
    """Translate the deployment hardware record into GenZ inputs."""

    hbm_efficiency = hardware.decode_gemm_mbu * 2039.0 / hardware.hbm_stream_gbps
    return RooflineHardware(
        peak_bf16_tflops=hardware.peak_bf16_tflops,
        compute_efficiency=hardware.bf16_compute_efficiency,
        hbm_stream_gbps=hardware.hbm_stream_gbps,
        hbm_kernel_efficiency=hbm_efficiency,
        fp8_dequant_gvalues_per_second=hardware.fp8_dequant_gbps,
        int4_dequant_gvalues_per_second=hardware.int4_dequant_gbps,
        nvlink_latency_ms=hardware.nvlink_latency_ms,
        nvlink_payload_gbps=hardware.nvlink_payload_gbps,
    )


@dataclass(frozen=True, slots=True)
class Scenario:
    """One model/context placement backed by a generated core profile."""

    id: str
    model_key: str
    model: str
    context_tokens: int
    max_users: int
    model_layers: int
    cache_layers: int
    cached_width: int
    pp_size: int
    tp_size: int
    nodes: int
    replicas: int
    weight_per_gpu_gb: float
    mtp_draft_tokens: int = 0

    @property
    def load_users(self) -> tuple[int, int, int, int, int]:
        return load_users_for_max(self.max_users)

    def core_tpot_for_sequences(self, sequences: int, hardware: A800Host | None = None) -> float:
        """Return batch latency from the generated LLMServingSim profile."""

        if sequences < 1:
            raise ValueError("sequences must be positive")
        if hardware is None:
            return llmservingsim_decode_ms(self.model_key, sequences)
        roofline = _roofline_hardware(hardware)
        if roofline == CENTRAL_HARDWARE:
            return llmservingsim_decode_ms(self.model_key, sequences)
        return analytical_decode_ms(self.model_key, sequences, roofline)

    def reference_tpot_for_users(self, users: int, hardware: A800Host | None = None) -> float:
        if users < 1 or users > self.max_users:
            raise ValueError("users must be between one and the admission ceiling")
        return self.core_tpot_for_sequences(math.ceil(users / self.replicas), hardware)

    def core_stage_tpot_for_sequences(
        self, sequences: int, hardware: A800Host | None = None
    ) -> tuple[float, ...]:
        """Return the per-stage profile used by the steady PP scheduler."""

        if sequences < 1:
            raise ValueError("sequences must be positive")
        if hardware is None:
            return llmservingsim_decode_stage_ms(self.model_key, sequences)
        roofline = _roofline_hardware(hardware)
        if roofline == CENTRAL_HARDWARE:
            return llmservingsim_decode_stage_ms(self.model_key, sequences)
        return analytical_decode_stage_ms(self.model_key, sequences, roofline)

    @property
    def ttft_seconds(self) -> tuple[float, float, float, float, float]:
        single = analytical_prefill_seconds(self.model_key, self.context_tokens)
        return cast(
            tuple[float, float, float, float, float],
            _burst_ttft(single, self.max_users, self.replicas)[0],
        )

    @property
    def last_ttft_seconds(self) -> tuple[float, float, float, float, float]:
        single = analytical_prefill_seconds(self.model_key, self.context_tokens)
        return cast(
            tuple[float, float, float, float, float],
            _burst_ttft(single, self.max_users, self.replicas)[1],
        )


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """An added event using LLMServingSim's per-stage trace vocabulary."""

    stage: int
    node: int
    name: str
    critical_ms: float
    background_ms: float = 0.0
    bytes_moved: float = 0.0


@dataclass(frozen=True, slots=True)
class PointEstimate:
    users: int
    users_per_replica: int
    selected_microbatch: int
    tpot_ms: float
    aggregate_tps: float
    per_user_tps: float
    sensitivity_p10_ms: float
    sensitivity_p90_ms: float
    trace_events: tuple[TraceEvent, ...]


@dataclass(frozen=True, slots=True)
class ResidencyPoint:
    requested_fraction: float
    resident_layers: int
    exact_fraction: float
    resident_gib_per_gpu: float
    hbm_feasible: bool
    estimate: PointEstimate


@dataclass(frozen=True, slots=True)
class NativeCacheLayer:
    """Growing native-cache and attention work for one transformer layer."""

    main_entries: int
    attended_entries: int
    index_entries: int = 0


def load_users_for_max(max_users: int) -> tuple[int, int, int, int, int]:
    """Return the report's single-user and 25/50/75/100% load points."""

    if max_users < 1:
        raise ValueError("admission ceiling must be positive")

    def half_up(value: float) -> int:
        return math.floor(value + 0.5)

    loads = tuple(max(1, half_up(max_users * load)) for load in (0.25, 0.5, 0.75, 1.0))
    return 1, loads[0], loads[1], loads[2], loads[3]


def _tracked_llmservingsim_root() -> Path:
    return Path(__file__).resolve().parents[2] / "extern" / "tracked" / "llmservingsim"


def _load_stage_boundary_function() -> Callable[[Sequence[int], int], list[int]]:
    """Load the pinned upstream partitioner instead of reimplementing PP splitting."""

    root = _tracked_llmservingsim_root()
    if not root.exists():
        raise RuntimeError("initialize extern/tracked/llmservingsim before simulation")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("serving.core.trace_generator")
    return cast(Callable[[Sequence[int], int], list[int]], module._pp_stage_boundaries)


def stage_layer_counts(num_layers: int, pp_size: int) -> tuple[int, ...]:
    """Return block counts using LLMServingSim's vLLM-compatible PP boundaries."""

    if num_layers <= 0 or pp_size <= 0:
        raise ValueError("layer and pipeline counts must be positive")
    boundaries = _load_stage_boundary_function()(list(range(num_layers)), pp_size)
    edges = (0, *boundaries, num_layers)
    return tuple(edges[index + 1] - edges[index] for index in range(pp_size))


def selected_entries(context_tokens: int, fraction: float = 1.0 / 64.0) -> int:
    if context_tokens <= 0 or not 0.0 < fraction <= 1.0:
        raise ValueError("invalid context length or selection fraction")
    return math.ceil(context_tokens * fraction)


def stored_bytes_per_value(storage: StorageName) -> float:
    """FP8 includes one four-byte scale per 128 stored values."""

    return 2.0 if storage == "bf16" else 1.0 + 4.0 / 128.0


def selector_ms_per_block(users: int, context_tokens: int) -> float:
    """Interpolate ShadowKV Table 13's landmark-selection measurements.

    Table 13 reports 0.80 ms at 24×128K and 0.88 ms at 12×256K for one
    Llama-3-8B-1M transformer block.  This intentionally remains an extrapolation
    when applied to the frontier-model indexers.
    """

    volume = users * context_tokens / (24 * 131_072)
    context_octaves = math.log2(context_tokens / 131_072)
    return max(0.02, 0.12 + 0.68 * volume + 0.08 * context_octaves)


def _paper_materialize_ms_per_block(
    *,
    users: int,
    context_tokens: int,
    cached_width: int,
    miss_fraction: float,
    storage: StorageName,
    dequant_gbps: float,
) -> tuple[float, float, float]:
    """Return (critical materialization, fetch, reconstruction) per block.

    ShadowKV Table 13 gives 1.66/1.36 ms for V fetch/K reconstruction at
    24×128K and a 40% miss set after its 60% temporal cache reuse.  Payload work
    scales with batch, selected tokens, and the frontier model's cached width.
    A small fixed kernel component is retained.  FP8 affects stored bytes, while
    reconstruction still produces BF16 operands and pays an explicit conversion.
    """

    if miss_fraction <= 0.0:
        return 0.0, 0.0, 0.0
    reference_missing = 0.40
    volume = users * context_tokens / (24 * 131_072)
    width_ratio = cached_width / 2048.0
    work = volume * width_ratio * miss_fraction / reference_missing
    context_octaves = math.log2(context_tokens / 131_072)
    byte_ratio = stored_bytes_per_value(storage) / 2.0
    fetch_ms = 0.18 + (1.48 + 0.09 * context_octaves) * work * byte_ratio
    reconstruct_ms = 0.22 + (1.14 + 0.13 * context_octaves) * work
    dequant_ms = 0.0
    if storage == "fp8":
        values = users * selected_entries(context_tokens) * cached_width * miss_fraction
        dequant_ms = values / (dequant_gbps * 1e9) * 1000.0
    return max(fetch_ms + dequant_ms, reconstruct_ms), fetch_ms, reconstruct_ms


def _stage_cache_bytes(
    scenario: Scenario,
    *,
    storage: StorageName,
) -> tuple[float, ...]:
    layer_counts = stage_layer_counts(scenario.cache_layers, scenario.pp_size)
    per_layer = (
        selected_entries(scenario.context_tokens)
        * scenario.cached_width
        * stored_bytes_per_value(storage)
        / scenario.tp_size
    )
    return tuple(count * per_layer for count in layer_counts)


def resident_layer_count(cache_layers: int, requested_fraction: float) -> int:
    """Quantize a requested cache fraction to whole cache-bearing layers."""

    if cache_layers <= 0 or not 0.0 <= requested_fraction <= 1.0:
        raise ValueError("invalid cache-layer count or resident fraction")
    return min(cache_layers, math.floor(cache_layers * requested_fraction + 0.5))


def residency_scan(cache_layers: int) -> tuple[tuple[float, int, float], ...]:
    """Return requested, whole-layer, and exact ratios for 0%, 10%, ..., 100%."""

    points = []
    for tenth in range(11):
        requested = tenth / 10.0
        layers = resident_layer_count(cache_layers, requested)
        points.append((requested, layers, layers / cache_layers))
    return tuple(points)


def _resident_counts_by_stage(layer_counts: Sequence[int], resident_layers: int) -> tuple[int, ...]:
    """Spread resident layers proportionally so one PP stage is not favored."""

    total_layers = sum(layer_counts)
    if not 0 <= resident_layers <= total_layers:
        raise ValueError("resident layer count is outside the model")
    raw = [resident_layers * count / total_layers for count in layer_counts]
    counts = [math.floor(value) for value in raw]
    remaining = resident_layers - sum(counts)
    order = sorted(range(len(raw)), key=lambda index: raw[index] - counts[index], reverse=True)
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(counts)


def resident_cache_gib_per_gpu(
    scenario: Scenario,
    *,
    storage: StorageName,
    users: int,
    resident_layers: int,
) -> float:
    """Return the maximum extra full-cache footprint on any stage GPU."""

    local_users = math.ceil(users / scenario.replicas)
    layer_counts = stage_layer_counts(scenario.cache_layers, scenario.pp_size)
    resident_counts = _resident_counts_by_stage(layer_counts, resident_layers)
    bytes_per_layer_gpu = (
        scenario.context_tokens
        * scenario.cached_width
        * stored_bytes_per_value(storage)
        * local_users
        / scenario.tp_size
    )
    return max(resident_counts, default=0) * bytes_per_layer_gpu / 2**30


@lru_cache(maxsize=128)
def native_cache_layers(scenario: Scenario) -> tuple[NativeCacheLayer, ...]:
    """Describe the official non-ShadowKV growing attention state.

    Dense MLA retains every latent KV entry. GLM's DSA retains exact latent KV
    for every attention layer and a shared 128-value index key only where its
    official ``indexer_types`` entry is ``full``. GLM Flash applies the same
    rule only to its 11 DSA layers and pools the index by four. DeepSeek V4
    Flash follows its official per-layer compression ratios and 128-token
    sliding window; ratio-4 layers also retain the source implementation's
    shared 128-value index key.
    """

    spec = MODEL_SPECS[scenario.model_key]
    context = scenario.context_tokens
    if spec.compress_ratios:
        layers = []
        for ratio in spec.compress_ratios[: scenario.model_layers]:
            compressed = math.ceil(context / ratio) if ratio else 0
            window = min(context, spec.sliding_window)
            main_entries = window + compressed
            if ratio == 4:
                attended = window + min(spec.index_topk, compressed)
                index_entries = compressed
            else:
                attended = main_entries
                index_entries = 0
            layers.append(NativeCacheLayer(main_entries, attended, index_entries))
        return tuple(layers)

    layers = []
    for index in range(scenario.model_layers):
        is_linear = bool(spec.layer_types) and spec.layer_types[index] == "linear_attention"
        if is_linear:
            layers.append(NativeCacheLayer(0, 0, 0))
            continue
        attended = context if not spec.index_topk else min(context, spec.index_topk)
        full_index = bool(spec.indexer_types) and spec.indexer_types[index] == "full"
        index_entries = math.ceil(context / spec.index_pool) if full_index else 0
        layers.append(NativeCacheLayer(context, attended, index_entries))
    return tuple(layers)


def _native_stage_layers(scenario: Scenario) -> tuple[tuple[NativeCacheLayer, ...], ...]:
    layers = native_cache_layers(scenario)
    counts = stage_layer_counts(scenario.model_layers, scenario.pp_size)
    stages = []
    start = 0
    for count in counts:
        stages.append(layers[start : start + count])
        start += count
    return tuple(stages)


def native_cache_gib_per_gpu(
    scenario: Scenario,
    *,
    storage: StorageName,
    users: int,
) -> float:
    """Maximum native full-cache allocation on one GPU across all PP stages."""

    if users < 1:
        raise ValueError("users must be positive")
    spec = MODEL_SPECS[scenario.model_key]
    local_users = math.ceil(users / scenario.replicas)
    stage_values = [
        sum(
            layer.main_entries * scenario.cached_width + layer.index_entries * spec.index_dim
            for layer in stage
        )
        for stage in _native_stage_layers(scenario)
    ]
    per_gpu_values = max(stage_values, default=0.0) * local_users / scenario.tp_size
    return per_gpu_values * stored_bytes_per_value(storage) / 2**30


def native_hbm_headroom_gib(
    scenario: Scenario,
    *,
    hbm_utilization: float = 0.90,
    runtime_reserve_gib: float = 6.0,
) -> float:
    """Planning HBM left after official-format weights and runtime reserve."""

    if not 0.0 < hbm_utilization <= 1.0 or runtime_reserve_gib < 0.0:
        raise ValueError("invalid HBM utilization or runtime reserve")
    weight_per_gpu_gib = scenario.weight_per_gpu_gb * 1e9 / 2**30
    return max(0.0, 80.0 * hbm_utilization - weight_per_gpu_gib - runtime_reserve_gib)


def no_shadowkv_max_users(
    scenario: Scenario,
    *,
    storage: StorageName,
    hbm_utilization: float = 0.90,
    runtime_reserve_gib: float = 6.0,
) -> int:
    """Memory-only admission ceiling for the full native cache in HBM."""

    per_replica_user = native_cache_gib_per_gpu(scenario, storage=storage, users=1)
    if per_replica_user <= 0.0:
        raise ValueError("native growing cache must consume positive HBM")
    per_replica = math.floor(
        native_hbm_headroom_gib(
            scenario,
            hbm_utilization=hbm_utilization,
            runtime_reserve_gib=runtime_reserve_gib,
        )
        / per_replica_user
    )
    return max(0, per_replica * scenario.replicas)


def _native_attention_stage_ms(
    scenario: Scenario,
    *,
    sequences: int,
    storage: StorageName,
    hardware: A800Host,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return native attention roofline time and cache bytes read per PP stage.

    Index scoring and exact/native-cache attention are causally sequential and
    therefore summed. Within each kernel, BF16 compute, HBM traffic, and fused
    FP8 conversion share a max roofline. The HBM denominator uses the measured
    representative kernel MBU rather than peak spec bandwidth.
    """

    if sequences < 1:
        raise ValueError("sequences must be positive")
    spec = MODEL_SPECS[scenario.model_key]
    bytes_per_value = stored_bytes_per_value(storage)
    effective_hbm_gbps = hardware.decode_gemm_mbu * 2039.0
    effective_compute_tflops = hardware.peak_bf16_tflops * hardware.bf16_compute_efficiency
    stage_times = []
    stage_bytes = []
    for stage in _native_stage_layers(scenario):
        main_values = (
            sequences
            * sum(layer.attended_entries * scenario.cached_width for layer in stage)
            / scenario.tp_size
        )
        main_flops = (
            2.0
            * sequences
            * sum(
                layer.attended_entries * spec.attention_heads * (spec.qk_head_dim + spec.v_head_dim)
                for layer in stage
            )
            / scenario.tp_size
        )
        main_bytes = main_values * bytes_per_value
        main_ms = max(
            main_flops / (effective_compute_tflops * 1e12) * 1000.0,
            main_bytes / (effective_hbm_gbps * 1e9) * 1000.0,
            (main_values / (hardware.fp8_dequant_gbps * 1e9) * 1000.0 if storage == "fp8" else 0.0),
        )

        index_values = (
            sequences
            * sum(layer.index_entries * spec.index_dim for layer in stage)
            / scenario.tp_size
        )
        index_flops = (
            2.0
            * sequences
            * sum(layer.index_entries * spec.index_heads * spec.index_dim for layer in stage)
            / scenario.tp_size
        )
        index_bytes = index_values * bytes_per_value
        index_ms = max(
            index_flops / (effective_compute_tflops * 1e12) * 1000.0,
            index_bytes / (effective_hbm_gbps * 1e9) * 1000.0,
            (
                index_values / (hardware.fp8_dequant_gbps * 1e9) * 1000.0
                if storage == "fp8"
                else 0.0
            ),
        )
        stage_times.append(main_ms + index_ms)
        stage_bytes.append(main_bytes + index_bytes)
    return tuple(stage_times), tuple(stage_bytes)


def _resident_materialize_ms_per_block(
    *,
    users: int,
    context_tokens: int,
    cached_width: int,
    miss_fraction: float,
    storage: StorageName,
    dequant_gbps: float,
) -> float:
    """Optional FP8 conversion when an exact full-layer KV cache is already in HBM.

    Resident layers retain exact K and V, so neither host fetch nor low-rank key
    reconstruction is needed after landmark selection.  The sparse HBM gather is part
    of the calibrated attention floor; only an explicit unfused FP8 conversion remains.
    """

    if storage == "bf16" or miss_fraction <= 0.0:
        return 0.0
    values = users * selected_entries(context_tokens) * cached_width * miss_fraction
    dequant_ms = values / (dequant_gbps * 1e9) * 1000.0
    return dequant_ms


def _transfer_time_ms(
    *,
    stage_bytes_per_gpu: float,
    factor: float,
    users: int,
    tp_size: int,
    hardware: A800Host,
) -> float:
    per_gpu = stage_bytes_per_gpu * factor * users / (hardware.pcie_per_gpu_gbps * 1e9)
    tp2_pair = (
        stage_bytes_per_gpu * tp_size * factor * users / (hardware.pcie_tp2_pair_gbps * 1e9)
        if tp_size == 2
        else 0.0
    )
    host = stage_bytes_per_gpu * tp_size * factor * users / (hardware.host_dram_gbps * 1e9)
    return max(per_gpu, tp2_pair, host) * 1000.0


def _estimate_once(
    scenario: Scenario,
    *,
    load_index: int,
    policy: PolicyName,
    storage: StorageName,
    oracle: OraclePrefetch,
    reuse: float,
    hardware: A800Host,
    selector_scale: float = 1.0,
    materialize_scale: float = 1.0,
    users_override: int | None = None,
    resident_layers: int = 0,
    mtp_accepted_tokens: int = 0,
    microbatch_override: int | None = None,
) -> tuple[float, tuple[TraceEvent, ...]]:
    users = scenario.load_users[load_index] if users_override is None else users_override
    if not 1 <= users <= scenario.max_users:
        raise ValueError("users must be within the scenario admission ceiling")
    if not 0 <= resident_layers <= scenario.cache_layers:
        raise ValueError("resident_layers must be within the cache-bearing layer count")
    if mtp_accepted_tokens < 0:
        raise ValueError("mtp_accepted_tokens cannot be negative")
    if mtp_accepted_tokens and scenario.mtp_draft_tokens == 0:
        raise ValueError(f"{scenario.model} has no checkpoint MTP component")
    if mtp_accepted_tokens > scenario.mtp_draft_tokens:
        raise ValueError("accepted MTP tokens cannot exceed the configured draft length")

    verification_tokens = scenario.mtp_draft_tokens if mtp_accepted_tokens else 1
    emitted_tokens = mtp_accepted_tokens + 1 if mtp_accepted_tokens else 1
    local_users = math.ceil(users / scenario.replicas)
    if microbatch_override is None:
        microbatch_override = _optimal_microbatch_size(
            scenario,
            users=users,
            policy=policy,
            storage=storage,
            oracle=oracle,
            reuse=reuse,
            resident_layers=resident_layers,
            mtp_accepted_tokens=mtp_accepted_tokens,
        )
    if not 1 <= microbatch_override <= local_users:
        raise ValueError("microbatch size must be within users per replica")
    batch_users = microbatch_override
    pipeline_groups = math.ceil(local_users / batch_users)
    layer_counts = stage_layer_counts(scenario.cache_layers, scenario.pp_size)
    resident_counts = _resident_counts_by_stage(layer_counts, resident_layers)
    stage_bytes = _stage_cache_bytes(scenario, storage=storage)

    # The profile contains only the model-forward path. ShadowKV work is added below,
    # so no transfer allowance is subtracted from this generated core floor.
    profile_floor = scenario.core_tpot_for_sequences(batch_users, hardware)
    verified_core_stages = scenario.core_stage_tpot_for_sequences(
        batch_users * verification_tokens, hardware
    )

    events: list[TraceEvent] = []
    selector_block = selector_ms_per_block(batch_users, scenario.context_tokens) * selector_scale
    if policy == "oracle-prefetch":
        prefetch_factor, jit_factor = oracle.traffic_factors(reuse=reuse)
        miss_fraction = jit_factor
    else:
        prefetch_factor = 0.0
        miss_fraction = 1.0 - reuse

    materialize_block, _, _ = _paper_materialize_ms_per_block(
        users=batch_users,
        context_tokens=scenario.context_tokens,
        cached_width=scenario.cached_width,
        miss_fraction=miss_fraction,
        storage=storage,
        dequant_gbps=hardware.fp8_dequant_gbps,
    )
    materialize_block *= materialize_scale
    resident_materialize_block = (
        _resident_materialize_ms_per_block(
            users=batch_users,
            context_tokens=scenario.context_tokens,
            cached_width=scenario.cached_width,
            miss_fraction=miss_fraction,
            storage=storage,
            dequant_gbps=hardware.fp8_dequant_gbps,
        )
        * materialize_scale
    )

    extra_selector_block = 0.0
    extra_materialize_block = 0.0
    extra_resident_materialize_block = 0.0
    extra_transfer_factor = 0.0
    if verification_tokens > 1:
        extra_users = batch_users * (verification_tokens - 1)
        extra_selector_block = (
            selector_ms_per_block(batch_users * verification_tokens, scenario.context_tokens)
            - selector_ms_per_block(batch_users, scenario.context_tokens)
        ) * selector_scale
        extra_materialize_block, _, _ = _paper_materialize_ms_per_block(
            users=extra_users,
            context_tokens=scenario.context_tokens,
            cached_width=scenario.cached_width,
            miss_fraction=1.0 - reuse,
            storage=storage,
            dequant_gbps=hardware.fp8_dequant_gbps,
        )
        extra_materialize_block *= materialize_scale
        extra_resident_materialize_block = (
            _resident_materialize_ms_per_block(
                users=extra_users,
                context_tokens=scenario.context_tokens,
                cached_width=scenario.cached_width,
                miss_fraction=1.0 - reuse,
                storage=storage,
                dequant_gbps=hardware.fp8_dequant_gbps,
            )
            * materialize_scale
        )
        extra_transfer_factor = (verification_tokens - 1) * (1.0 - reuse)

    stage_critical_times: list[float] = []
    for stage, (layers, resident, bytes_for_stage) in enumerate(
        zip(layer_counts, resident_counts, stage_bytes, strict=True)
    ):
        node = min(scenario.nodes - 1, stage * scenario.nodes // scenario.pp_size)
        offloaded = layers - resident
        must_select = policy == "fetch-at-decode" or oracle.verify_with_landmarks
        selection = layers * selector_block if must_select else 0.0
        materialize = offloaded * materialize_block + resident * resident_materialize_block
        transfer = _transfer_time_ms(
            stage_bytes_per_gpu=bytes_for_stage * offloaded / layers,
            factor=miss_fraction,
            users=batch_users,
            tp_size=scenario.tp_size,
            hardware=hardware,
        )
        # Table 13's materialization calibration already includes transfer.  The
        # physical-link result is a lower bound that takes over when larger than it.
        materialize = max(materialize, transfer)
        extra_selection = layers * extra_selector_block
        extra_materialize = (
            offloaded * extra_materialize_block + resident * extra_resident_materialize_block
        )
        extra_transfer = _transfer_time_ms(
            stage_bytes_per_gpu=bytes_for_stage * offloaded / layers,
            factor=extra_transfer_factor,
            users=batch_users,
            tp_size=scenario.tp_size,
            hardware=hardware,
        )
        extra_materialize = max(extra_materialize, extra_transfer)
        stage_critical = selection + materialize + extra_selection + extra_materialize
        stage_critical_times.append(stage_critical)
        events.append(
            TraceEvent(
                stage=stage,
                node=node,
                name=(
                    "mtp_verify+landmark_select+miss_materialize"
                    if verification_tokens > 1
                    else "landmark_select+miss_materialize"
                ),
                critical_ms=stage_critical,
                bytes_moved=(
                    bytes_for_stage
                    * offloaded
                    / layers
                    * (miss_fraction + extra_transfer_factor)
                    * batch_users
                ),
            )
        )

    background_stall = 0.0
    if prefetch_factor > 0.0:
        node_bytes = [0.0] * scenario.nodes
        per_gpu_times: list[float] = []
        tp2_pair_times: list[float] = []
        for stage, (layers, resident, bytes_for_stage) in enumerate(
            zip(layer_counts, resident_counts, stage_bytes, strict=True)
        ):
            node = min(scenario.nodes - 1, stage * scenario.nodes // scenario.pp_size)
            moved = bytes_for_stage * (layers - resident) / layers * prefetch_factor * batch_users
            node_bytes[node] += moved * scenario.tp_size
            per_gpu_times.append(moved / (hardware.pcie_per_gpu_gbps * 1e9) * 1000.0)
            if scenario.tp_size == 2:
                tp2_pair_times.append(
                    moved * scenario.tp_size / (hardware.pcie_tp2_pair_gbps * 1e9) * 1000.0
                )
            events.append(
                TraceEvent(
                    stage=stage,
                    node=node,
                    name="oracle_prefetch",
                    critical_ms=0.0,
                    background_ms=per_gpu_times[-1],
                    bytes_moved=moved,
                )
            )
        background_service = max(
            max(per_gpu_times, default=0.0),
            max(tp2_pair_times, default=0.0),
            max(
                (value / (hardware.host_dram_gbps * 1e9) * 1000.0 for value in node_bytes),
                default=0.0,
            ),
        )
        # A steady PP schedule admits one microbatch per stage cadence. Across
        # one recurrence, every request group contributes one prefetch payload.
        # The transfer can overlap compute, but its node service demand must fit.
        background_stall = background_service * pipeline_groups / oracle.lookahead_tokens

    stage_communications = [0.0] * scenario.pp_size
    if scenario.pp_size > 1:
        for stage in range(scenario.pp_size - 1):
            source_node = min(scenario.nodes - 1, stage * scenario.nodes // scenario.pp_size)
            target_node = min(scenario.nodes - 1, (stage + 1) * scenario.nodes // scenario.pp_size)
            stage_communications[stage] = (
                hardware.ib_latency_ms if source_node != target_node else hardware.nvlink_latency_ms
            )
    mtp_draft = 0.0
    if verification_tokens > 1:
        # One checkpoint MTP block per config. Its active path is approximated by
        # one full-model layer, while the target verification itself comes from the
        # generated profile at the enlarged token batch above.
        draft_core = profile_floor * verification_tokens / scenario.model_layers
        draft_materialize, _, _ = _paper_materialize_ms_per_block(
            users=batch_users,
            context_tokens=scenario.context_tokens,
            cached_width=scenario.cached_width,
            miss_fraction=1.0 - reuse,
            storage=storage,
            dequant_gbps=hardware.fp8_dequant_gbps,
        )
        draft_block = selector_block + draft_materialize * materialize_scale
        mtp_draft = draft_core + verification_tokens * draft_block
        events.append(
            TraceEvent(
                stage=scenario.pp_size - 1,
                node=scenario.nodes - 1,
                name="mtp_autoregressive_draft",
                critical_ms=mtp_draft,
            )
        )

    stage_critical_times[-1] += mtp_draft
    stage_services = [
        core + shadow + communication
        for core, shadow, communication in zip(
            verified_core_stages, stage_critical_times, stage_communications, strict=True
        )
    ]
    # LLMServingSim caps in-flight batches at PP depth. With fewer groups, a
    # group's next token waits for its own P-stage traversal; with more groups,
    # queued groups determine the recurrence. The scheduler selects the central
    # microbatch size that minimizes this steady-state recurrence.
    compute_recurrence = max(scenario.pp_size, pipeline_groups) * max(stage_services)
    round_recurrence = max(compute_recurrence, background_stall)
    return round_recurrence / emitted_tokens, tuple(events)


DEFAULT_ORACLE = OraclePrefetch()
DEFAULT_HARDWARE = A800Host()


@lru_cache(maxsize=8192)
def _optimal_microbatch_size(
    scenario: Scenario,
    *,
    users: int,
    policy: PolicyName,
    storage: StorageName,
    oracle: OraclePrefetch,
    reuse: float,
    resident_layers: int,
    mtp_accepted_tokens: int,
) -> int:
    """Choose the central steady-state PP microbatch size.

    There is no advantage to using fewer groups than PP stages: it leaves
    stages idle while increasing each stage's batch work. Therefore the
    bounded search ends at ``ceil(users_per_replica / pp_size)``. PP=1 keeps
    the entire continuous batch together.
    """

    local_users = math.ceil(users / scenario.replicas)
    if scenario.pp_size == 1:
        return local_users
    maximum = math.ceil(local_users / scenario.pp_size)
    candidates = range(1, maximum + 1)
    return min(
        candidates,
        key=lambda microbatch: _estimate_once(
            scenario,
            load_index=0,
            policy=policy,
            storage=storage,
            oracle=oracle,
            reuse=reuse,
            hardware=DEFAULT_HARDWARE,
            users_override=users,
            resident_layers=resident_layers,
            mtp_accepted_tokens=mtp_accepted_tokens,
            microbatch_override=microbatch,
        )[0],
    )


def estimate_point(
    scenario: Scenario,
    *,
    load_index: int,
    policy: PolicyName,
    storage: StorageName,
    oracle: OraclePrefetch = DEFAULT_ORACLE,
    reuse: float = 0.60,
    hardware: A800Host = DEFAULT_HARDWARE,
    sensitivity_samples: int = 256,
    seed: int = 17,
    resident_layers: int = 0,
    mtp_accepted_tokens: int = 0,
    users_override: int | None = None,
) -> PointEstimate:
    """Estimate one operating point plus a deterministic parameter-sensitivity band."""

    central, events = _estimate_once(
        scenario,
        load_index=load_index,
        policy=policy,
        storage=storage,
        oracle=oracle,
        reuse=reuse,
        hardware=hardware,
        users_override=users_override,
        resident_layers=resident_layers,
        mtp_accepted_tokens=mtp_accepted_tokens,
    )
    rng = random.Random(f"{seed}:{scenario.id}:{load_index}:{policy}:{storage}")
    samples: list[float] = []
    for _ in range(max(0, sensitivity_samples)):
        sample_hardware = A800Host(
            hbm_stream_gbps=rng.triangular(1700.0, 1810.0, hardware.hbm_stream_gbps),
            decode_gemm_mbu=rng.triangular(0.307, 0.618, hardware.decode_gemm_mbu),
            peak_bf16_tflops=hardware.peak_bf16_tflops,
            bf16_compute_efficiency=rng.triangular(0.30, 0.55, hardware.bf16_compute_efficiency),
            pcie_per_gpu_gbps=rng.triangular(20.5, 23.5, hardware.pcie_per_gpu_gbps),
            pcie_tp2_pair_gbps=rng.triangular(23.0, 28.0, hardware.pcie_tp2_pair_gbps),
            host_dram_gbps=rng.triangular(130.0, 220.0, hardware.host_dram_gbps),
            fp8_dequant_gbps=rng.triangular(320.0, 520.0, hardware.fp8_dequant_gbps),
            int4_dequant_gbps=rng.triangular(280.0, 520.0, hardware.int4_dequant_gbps),
            nvlink_latency_ms=hardware.nvlink_latency_ms,
            nvlink_payload_gbps=rng.triangular(240.0, 300.0, hardware.nvlink_payload_gbps),
            ib_latency_ms=hardware.ib_latency_ms,
            ib_payload_gbps=hardware.ib_payload_gbps,
        )
        value, _ = _estimate_once(
            scenario,
            load_index=load_index,
            policy=policy,
            storage=storage,
            oracle=oracle,
            reuse=reuse,
            hardware=sample_hardware,
            selector_scale=rng.lognormvariate(-0.5 * 0.20**2, 0.20),
            materialize_scale=rng.lognormvariate(-0.5 * 0.25**2, 0.25),
            users_override=users_override,
            resident_layers=resident_layers,
            mtp_accepted_tokens=mtp_accepted_tokens,
        )
        samples.append(value)
    samples.sort()

    def percentile(values: Sequence[float], fraction: float) -> float:
        if not values:
            return central
        return values[min(len(values) - 1, round((len(values) - 1) * fraction))]

    users = scenario.load_users[load_index] if users_override is None else users_override
    aggregate = 1000.0 * users / central
    selected_microbatch = _optimal_microbatch_size(
        scenario,
        users=users,
        policy=policy,
        storage=storage,
        oracle=oracle,
        reuse=reuse,
        resident_layers=resident_layers,
        mtp_accepted_tokens=mtp_accepted_tokens,
    )
    return PointEstimate(
        users=users,
        users_per_replica=math.ceil(users / scenario.replicas),
        selected_microbatch=selected_microbatch,
        tpot_ms=central,
        aggregate_tps=aggregate,
        per_user_tps=1000.0 / central,
        sensitivity_p10_ms=percentile(samples, 0.10),
        sensitivity_p90_ms=percentile(samples, 0.90),
        trace_events=events,
    )


def _estimate_no_shadowkv_once(
    scenario: Scenario,
    *,
    users: int,
    storage: StorageName,
    hardware: A800Host,
    mtp_accepted_tokens: int = 0,
    microbatch_override: int | None = None,
) -> tuple[float, tuple[TraceEvent, ...]]:
    """Estimate full native-cache serving with no ShadowKV or host offload."""

    maximum = no_shadowkv_max_users(scenario, storage=storage)
    if not 1 <= users <= maximum:
        raise ValueError("users must fit the no-ShadowKV HBM admission ceiling")
    if mtp_accepted_tokens < 0:
        raise ValueError("mtp_accepted_tokens cannot be negative")
    if mtp_accepted_tokens and scenario.mtp_draft_tokens == 0:
        raise ValueError(f"{scenario.model} has no checkpoint MTP component")
    if mtp_accepted_tokens > scenario.mtp_draft_tokens:
        raise ValueError("accepted MTP tokens cannot exceed the configured draft length")

    verification_tokens = scenario.mtp_draft_tokens if mtp_accepted_tokens else 1
    emitted_tokens = mtp_accepted_tokens + 1 if mtp_accepted_tokens else 1
    local_users = math.ceil(users / scenario.replicas)
    if microbatch_override is None:
        microbatch_override = _optimal_no_shadowkv_microbatch_size(
            scenario,
            users=users,
            storage=storage,
            mtp_accepted_tokens=mtp_accepted_tokens,
        )
    if not 1 <= microbatch_override <= local_users:
        raise ValueError("microbatch size must be within users per replica")
    batch_users = microbatch_override
    pipeline_groups = math.ceil(local_users / batch_users)
    verified_sequences = batch_users * verification_tokens
    core_stages = scenario.core_stage_tpot_for_sequences(verified_sequences, hardware)
    attention_stages, cache_bytes = _native_attention_stage_ms(
        scenario,
        sequences=verified_sequences,
        storage=storage,
        hardware=hardware,
    )

    stage_communications = [0.0] * scenario.pp_size
    if scenario.pp_size > 1:
        for stage in range(scenario.pp_size - 1):
            source_node = min(scenario.nodes - 1, stage * scenario.nodes // scenario.pp_size)
            target_node = min(scenario.nodes - 1, (stage + 1) * scenario.nodes // scenario.pp_size)
            stage_communications[stage] = (
                hardware.ib_latency_ms if source_node != target_node else hardware.nvlink_latency_ms
            )

    events = [
        TraceEvent(
            stage=stage,
            node=min(scenario.nodes - 1, stage * scenario.nodes // scenario.pp_size),
            name="native_attention_hbm",
            critical_ms=attention_ms,
            bytes_moved=bytes_read,
        )
        for stage, (attention_ms, bytes_read) in enumerate(
            zip(attention_stages, cache_bytes, strict=True)
        )
    ]

    mtp_draft = 0.0
    if verification_tokens > 1:
        one_core = scenario.core_tpot_for_sequences(batch_users, hardware)
        one_attention, _ = _native_attention_stage_ms(
            scenario,
            sequences=batch_users,
            storage=storage,
            hardware=hardware,
        )
        mtp_draft = verification_tokens * (
            one_core / scenario.model_layers + sum(one_attention) / scenario.model_layers
        )
        events.append(
            TraceEvent(
                stage=scenario.pp_size - 1,
                node=scenario.nodes - 1,
                name="mtp_autoregressive_draft",
                critical_ms=mtp_draft,
            )
        )

    stage_services = [
        core + attention + communication
        for core, attention, communication in zip(
            core_stages, attention_stages, stage_communications, strict=True
        )
    ]
    stage_services[-1] += mtp_draft
    recurrence = max(scenario.pp_size, pipeline_groups) * max(stage_services)
    return recurrence / emitted_tokens, tuple(events)


@lru_cache(maxsize=4096)
def _optimal_no_shadowkv_microbatch_size(
    scenario: Scenario,
    *,
    users: int,
    storage: StorageName,
    mtp_accepted_tokens: int,
) -> int:
    """Choose the throughput-optimal PP microbatch for native-cache serving."""

    local_users = math.ceil(users / scenario.replicas)
    if scenario.pp_size == 1:
        return local_users
    maximum = math.ceil(local_users / scenario.pp_size)
    return min(
        range(1, maximum + 1),
        key=lambda microbatch: _estimate_no_shadowkv_once(
            scenario,
            users=users,
            storage=storage,
            hardware=DEFAULT_HARDWARE,
            mtp_accepted_tokens=mtp_accepted_tokens,
            microbatch_override=microbatch,
        )[0],
    )


def estimate_no_shadowkv_point(
    scenario: Scenario,
    *,
    users: int,
    storage: StorageName,
    hardware: A800Host = DEFAULT_HARDWARE,
    sensitivity_samples: int = 256,
    seed: int = 17,
    mtp_accepted_tokens: int = 0,
) -> PointEstimate:
    """Estimate a no-offload native-cache point within its HBM admission cap."""

    microbatch = _optimal_no_shadowkv_microbatch_size(
        scenario,
        users=users,
        storage=storage,
        mtp_accepted_tokens=mtp_accepted_tokens,
    )
    central, events = _estimate_no_shadowkv_once(
        scenario,
        users=users,
        storage=storage,
        hardware=hardware,
        mtp_accepted_tokens=mtp_accepted_tokens,
        microbatch_override=microbatch,
    )
    rng = random.Random(f"{seed}:{scenario.id}:no-shadowkv:{users}:{storage}")
    samples = []
    for _ in range(max(0, sensitivity_samples)):
        sample_hardware = A800Host(
            hbm_stream_gbps=rng.triangular(1700.0, 1810.0, hardware.hbm_stream_gbps),
            decode_gemm_mbu=rng.triangular(0.307, 0.618, hardware.decode_gemm_mbu),
            peak_bf16_tflops=hardware.peak_bf16_tflops,
            bf16_compute_efficiency=rng.triangular(0.30, 0.55, hardware.bf16_compute_efficiency),
            pcie_per_gpu_gbps=hardware.pcie_per_gpu_gbps,
            pcie_tp2_pair_gbps=hardware.pcie_tp2_pair_gbps,
            host_dram_gbps=hardware.host_dram_gbps,
            fp8_dequant_gbps=rng.triangular(320.0, 520.0, hardware.fp8_dequant_gbps),
            int4_dequant_gbps=rng.triangular(280.0, 520.0, hardware.int4_dequant_gbps),
            nvlink_latency_ms=hardware.nvlink_latency_ms,
            nvlink_payload_gbps=rng.triangular(240.0, 300.0, hardware.nvlink_payload_gbps),
            ib_latency_ms=hardware.ib_latency_ms,
            ib_payload_gbps=hardware.ib_payload_gbps,
        )
        sample, _ = _estimate_no_shadowkv_once(
            scenario,
            users=users,
            storage=storage,
            hardware=sample_hardware,
            mtp_accepted_tokens=mtp_accepted_tokens,
            microbatch_override=microbatch,
        )
        samples.append(sample)
    samples.sort()

    def percentile(fraction: float) -> float:
        if not samples:
            return central
        return samples[min(len(samples) - 1, round((len(samples) - 1) * fraction))]

    return PointEstimate(
        users=users,
        users_per_replica=math.ceil(users / scenario.replicas),
        selected_microbatch=microbatch,
        tpot_ms=central,
        aggregate_tps=1000.0 * users / central,
        per_user_tps=1000.0 / central,
        sensitivity_p10_ms=percentile(0.10),
        sensitivity_p90_ms=percentile(0.90),
        trace_events=events,
    )


def estimate_residency_scan(
    scenario: Scenario,
    *,
    users: int,
    policy: PolicyName,
    storage: StorageName,
    oracle: OraclePrefetch = DEFAULT_ORACLE,
    reuse: float = 0.60,
    hardware: A800Host = DEFAULT_HARDWARE,
    hbm_utilization: float = 0.90,
    runtime_reserve_gib: float = 6.0,
) -> tuple[ResidencyPoint, ...]:
    """Evaluate whole-layer KV residency at nominal ten-percentage-point steps."""

    if not 0.0 < hbm_utilization <= 1.0 or runtime_reserve_gib < 0.0:
        raise ValueError("invalid HBM utilization or runtime reserve")
    weight_per_gpu_gib = scenario.weight_per_gpu_gb * 1e9 / 2**30
    headroom = 80.0 * hbm_utilization - weight_per_gpu_gib - runtime_reserve_gib
    points = []
    for requested, layers, exact in residency_scan(scenario.cache_layers):
        resident_gib = resident_cache_gib_per_gpu(
            scenario,
            storage=storage,
            users=users,
            resident_layers=layers,
        )
        estimate = estimate_point(
            scenario,
            load_index=0,
            policy=policy,
            storage=storage,
            oracle=oracle,
            reuse=reuse,
            hardware=hardware,
            sensitivity_samples=0,
            users_override=users,
            resident_layers=layers,
        )
        points.append(
            ResidencyPoint(
                requested_fraction=requested,
                resident_layers=layers,
                exact_fraction=exact,
                resident_gib_per_gpu=resident_gib,
                hbm_feasible=resident_gib <= max(0.0, headroom),
                estimate=estimate,
            )
        )
    return tuple(points)


def _burst_ttft(
    single_seconds: float, max_users: int, replicas: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    users = (1, *(max(1, math.floor(max_users * load + 0.5)) for load in (0.25, 0.5, 0.75, 1.0)))
    averages = []
    lasts = []
    for active in users:
        local = math.ceil(active / replicas)
        averages.append(single_seconds * (local + 1) / 2)
        lasts.append(single_seconds * local)
    return tuple(averages), tuple(lasts)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("kimi-72", "kimi", "Kimi Code 2.7", 73_728, 48, 61, 61, 576, 8, 2, 2, 1, 37.2, 0),
    Scenario("kimi-128", "kimi", "Kimi Code 2.7", 131_072, 28, 61, 61, 576, 8, 2, 2, 1, 37.2, 0),
    Scenario("kimi-256", "kimi", "Kimi Code 2.7", 262_144, 14, 61, 61, 576, 8, 2, 2, 1, 37.2, 0),
    Scenario("glm-72", "glm", "GLM-5.3", 73_728, 48, 78, 78, 576, 8, 2, 2, 1, 47.225, 3),
    Scenario("glm-128", "glm", "GLM-5.3", 131_072, 40, 78, 78, 576, 8, 2, 2, 1, 47.225, 3),
    Scenario("glm-256", "glm", "GLM-5.3", 262_144, 30, 78, 78, 576, 8, 2, 2, 1, 47.225, 3),
    Scenario(
        "glm-flash-72",
        "glm-flash",
        "GLM-5.3-Flash",
        73_728,
        80,
        45,
        11,
        512,
        1,
        8,
        1,
        1,
        328.326771576 / 8,
        2,
    ),
    Scenario(
        "glm-flash-128",
        "glm-flash",
        "GLM-5.3-Flash",
        131_072,
        64,
        45,
        11,
        512,
        1,
        8,
        1,
        1,
        328.326771576 / 8,
        2,
    ),
    Scenario(
        "glm-flash-256",
        "glm-flash",
        "GLM-5.3-Flash",
        262_144,
        48,
        45,
        11,
        512,
        1,
        8,
        1,
        1,
        328.326771576 / 8,
        2,
    ),
    Scenario(
        "flash-72",
        "deepseek-flash",
        "DeepSeek V4 Flash",
        73_728,
        80,
        43,
        21,
        512,
        1,
        4,
        1,
        2,
        39.9,
        2,
    ),
    Scenario(
        "flash-128",
        "deepseek-flash",
        "DeepSeek V4 Flash",
        131_072,
        64,
        43,
        21,
        512,
        1,
        4,
        1,
        2,
        39.9,
        2,
    ),
    Scenario(
        "flash-256",
        "deepseek-flash",
        "DeepSeek V4 Flash",
        262_144,
        48,
        43,
        21,
        512,
        1,
        4,
        1,
        2,
        39.9,
        2,
    ),
)


def build_report(
    *,
    sensitivity_samples: int = 256,
    oracle: OraclePrefetch = DEFAULT_ORACLE,
    reuse: float = 0.60,
) -> dict[str, Any]:
    """Build the machine-readable dataset used by the interactive report."""

    series: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        item: dict[str, Any] = {
            "id": scenario.id,
            "model": scenario.model,
            "context_tokens": scenario.context_tokens,
            "max_users": scenario.max_users,
            "load_users": scenario.load_users,
            "stage_layer_counts": stage_layer_counts(scenario.cache_layers, scenario.pp_size),
            "ttft_seconds": scenario.ttft_seconds,
            "last_ttft_seconds": scenario.last_ttft_seconds,
            "core_profile_ms": tuple(
                scenario.reference_tpot_for_users(users) for users in scenario.load_users
            ),
            "core_single_user_breakdown": decode_breakdown(scenario.model_key, 1),
            "results": {},
        }
        for storage in ("bf16", "fp8"):
            storage_results: dict[str, Any] = {}
            for policy in ("oracle-prefetch", "fetch-at-decode"):
                points = [
                    estimate_point(
                        scenario,
                        load_index=index,
                        policy=policy,
                        storage=storage,
                        oracle=oracle,
                        reuse=reuse,
                        sensitivity_samples=sensitivity_samples,
                    )
                    for index in range(5)
                ]
                storage_results[policy] = [asdict(point) for point in points]
            native_max = no_shadowkv_max_users(scenario, storage=storage)
            if native_max < 1:
                raise RuntimeError(f"{scenario.id} cannot admit one native-cache request")
            native_users = load_users_for_max(native_max)
            storage_results["no-shadowkv"] = {
                "max_users": native_max,
                "load_users": native_users,
                "native_cache_gib_per_gpu_per_user": native_cache_gib_per_gpu(
                    scenario, storage=storage, users=1
                ),
                "hbm_headroom_gib": native_hbm_headroom_gib(scenario),
                "points": [
                    asdict(
                        estimate_no_shadowkv_point(
                            scenario,
                            users=users,
                            storage=storage,
                            sensitivity_samples=sensitivity_samples,
                        )
                    )
                    for users in native_users
                ],
            }
            cast(dict[str, Any], item["results"])[storage] = storage_results
        series.append(item)
    return {
        "schema_version": 1,
        "engine": {
            "name": "GenZ roofline → LLMServingSim profile tables → ShadowKV trace extension",
            "genz_commit": GENZ_COMMIT,
            "upstream_commit": LLMSERVINGSIM_COMMIT,
            "integration": (
                "official-config GenZ operators written to per_sequence.csv; "
                "LLMServingSim _lookup_per_sequence and _pp_stage_boundaries; "
                "external per-block ShadowKV events"
            ),
        },
        "shadowkv": {
            "selection_fraction": 1.0 / 64.0,
            "temporal_reuse": reuse,
            "oracle_recall": oracle.recall,
            "oracle_precision": oracle.precision,
            "oracle_lookahead_tokens": oracle.lookahead_tokens,
            "selection_verified_at_decode": oracle.verify_with_landmarks,
        },
        "sensitivity": {
            "samples": sensitivity_samples,
            "interval": "p10-p90 parameter sensitivity; not a statistical confidence interval",
        },
        "hardware_calibration": {
            "source": "synthetic A100-SXM4-80GB GPUs 2+3; used as an Ampere proxy for A800",
            "artifact": "benchmarks/a100-sxm4/results/a100-sxm4-80gb-gpu2-3.json",
            "hbm_stream_gbps": DEFAULT_HARDWARE.hbm_stream_gbps,
            "representative_bf16_gemm_mbu": DEFAULT_HARDWARE.decode_gemm_mbu,
            "bf16_compute_efficiency": DEFAULT_HARDWARE.bf16_compute_efficiency,
            "h2d_single_gpu_gbps": DEFAULT_HARDWARE.pcie_per_gpu_gbps,
            "h2d_concurrent_tp2_pair_gbps": DEFAULT_HARDWARE.pcie_tp2_pair_gbps,
            "fp8_dequant_gvalues_per_second": DEFAULT_HARDWARE.fp8_dequant_gbps,
            "model_core_floor": "replaced by GenZ-generated LLMServingSim profile rows",
            "profile_manifest": profile_manifest(),
        },
        "series": series,
    }


def _compact_point(point: PointEstimate) -> dict[str, float | int]:
    return {
        "users": point.users,
        "microbatch": point.selected_microbatch,
        "tpot": round(point.tpot_ms, 4),
        "tps": round(point.aggregate_tps, 4),
        "userTps": round(point.per_user_tps, 4),
        "p10": round(point.sensitivity_p10_ms, 4),
        "p90": round(point.sensitivity_p90_ms, 4),
    }


def build_interactive_dataset(*, sensitivity_samples: int = 256) -> dict[str, Any]:
    """Build the compact inline dataset used by the standalone HTML report."""

    policy_names: tuple[tuple[str, PolicyName], ...] = (
        ("ahead", "oracle-prefetch"),
        ("fetch", "fetch-at-decode"),
    )
    residency_loads = (0, *range(10, 101, 10))
    series = []
    for scenario in SCENARIOS:
        item: dict[str, Any] = {
            "id": scenario.id,
            "model": scenario.model,
            "context": scenario.context_tokens // 1024,
            "max": scenario.max_users,
            "mtpDraft": scenario.mtp_draft_tokens,
            "ttft": scenario.ttft_seconds,
            "last": scenario.last_ttft_seconds,
            "core": [
                round(scenario.reference_tpot_for_users(users), 4) for users in scenario.load_users
            ],
            "base": {},
            "mtp": {},
            "residency": {},
            "admission": {},
        }
        for storage in ("bf16", "fp8"):
            base_storage: dict[str, Any] = {}
            mtp_storage: dict[str, Any] = {}
            residency_storage: dict[str, Any] = {}
            for short_policy, policy in policy_names:
                base_storage[short_policy] = [
                    _compact_point(
                        estimate_point(
                            scenario,
                            load_index=index,
                            policy=policy,
                            storage=storage,
                            sensitivity_samples=sensitivity_samples,
                        )
                    )
                    for index in range(5)
                ]
                if scenario.mtp_draft_tokens:
                    mtp_storage[short_policy] = {
                        str(accepted): [
                            _compact_point(
                                estimate_point(
                                    scenario,
                                    load_index=index,
                                    policy=policy,
                                    storage=storage,
                                    sensitivity_samples=0,
                                    mtp_accepted_tokens=accepted,
                                )
                            )
                            for index in range(5)
                        ]
                        for accepted in (1, 2)
                    }
                residency_curves = []
                for load in residency_loads:
                    users = (
                        1
                        if load == 0
                        else max(1, math.floor(scenario.max_users * load / 100 + 0.5))
                    )
                    points = estimate_residency_scan(
                        scenario,
                        users=users,
                        policy=policy,
                        storage=storage,
                    )
                    residency_curves.append(
                        {
                            "load": load,
                            "users": users,
                            "points": [
                                {
                                    "requested": round(point.requested_fraction * 100, 4),
                                    "actual": round(point.exact_fraction * 100, 4),
                                    "layers": point.resident_layers,
                                    "residentGiB": round(point.resident_gib_per_gpu, 4),
                                    "feasible": point.hbm_feasible,
                                    **_compact_point(point.estimate),
                                }
                                for point in points
                            ],
                        }
                    )
                residency_storage[short_policy] = residency_curves
            native_max = no_shadowkv_max_users(scenario, storage=storage)
            if native_max < 1:
                raise RuntimeError(f"{scenario.id} cannot admit one native-cache request")
            native_users = load_users_for_max(native_max)
            base_storage["noShadow"] = [
                _compact_point(
                    estimate_no_shadowkv_point(
                        scenario,
                        users=users,
                        storage=storage,
                        sensitivity_samples=sensitivity_samples,
                    )
                )
                for users in native_users
            ]
            if scenario.mtp_draft_tokens:
                mtp_storage["noShadow"] = {
                    str(accepted): [
                        _compact_point(
                            estimate_no_shadowkv_point(
                                scenario,
                                users=users,
                                storage=storage,
                                sensitivity_samples=0,
                                mtp_accepted_tokens=accepted,
                            )
                        )
                        for users in native_users
                    ]
                    for accepted in (1, 2)
                }
            single_prefill = analytical_prefill_seconds(scenario.model_key, scenario.context_tokens)
            native_ttft, native_last = _burst_ttft(single_prefill, native_max, scenario.replicas)
            cast(dict[str, Any], item["admission"])[storage] = {
                "noShadow": {
                    "max": native_max,
                    "users": native_users,
                    "ttft": native_ttft,
                    "last": native_last,
                    "cacheGiBPerGpuPerUser": round(
                        native_cache_gib_per_gpu(scenario, storage=storage, users=1), 6
                    ),
                    "hbmHeadroomGiB": round(native_hbm_headroom_gib(scenario), 6),
                }
            }
            cast(dict[str, Any], item["base"])[storage] = base_storage
            cast(dict[str, Any], item["mtp"])[storage] = mtp_storage
            cast(dict[str, Any], item["residency"])[storage] = residency_storage
        series.append(item)
    return {
        "schemaVersion": 5,
        "loads": [0, 25, 50, 75, 100],
        "residencyLoads": list(residency_loads),
        "residencyRequested": list(range(0, 101, 10)),
        "series": series,
        "assumptions": {
            "mtpAcceptedPrefixTokens": [1, 2],
            "mtpExtraCandidatesUseDecodeFetch": True,
            "hbmUtilization": 0.90,
            "runtimeReserveGiB": 6.0,
            "residentLayersAreBalancedAcrossStages": True,
            "noShadowKVAdmissionIsMemoryOnly": True,
            "noShadowKVRejectsBeyondAdmission": True,
            "noShadowKVNativeIndexCacheIsSharedAcrossHeads": True,
            "hardwareCalibration": {
                "source": "synthetic A100-SXM4-80GB GPUs 2+3, used as an A800 proxy",
                "artifact": "benchmarks/a100-sxm4/results/a100-sxm4-80gb-gpu2-3.json",
                "hbmStreamGBps": DEFAULT_HARDWARE.hbm_stream_gbps,
                "representativeBF16GemmMBU": DEFAULT_HARDWARE.decode_gemm_mbu,
                "bf16ComputeEfficiency": DEFAULT_HARDWARE.bf16_compute_efficiency,
                "h2dSingleGPU_GBps": DEFAULT_HARDWARE.pcie_per_gpu_gbps,
                "h2dConcurrentTP2Pair_GBps": DEFAULT_HARDWARE.pcie_tp2_pair_gbps,
                "fp8DequantGvaluesPerSecond": DEFAULT_HARDWARE.fp8_dequant_gbps,
                "modelCoreFloorReplaced": True,
                "profileManifest": profile_manifest(),
            },
        },
    }


def concise_table(report: dict[str, Any], *, storage: StorageName = "fp8") -> str:
    """Return a compact Markdown table of 50% and 100% operating points."""

    lines = [
        "| Scenario | Oracle 50% / 100% total-per-user tok/s | Fetch 50% / 100% total-per-user tok/s |",
        "|---|---:|---:|",
    ]
    for item in cast(Iterable[dict[str, Any]], report["series"]):
        results = cast(dict[str, Any], item["results"])[storage]

        def cell(policy: str, result_set: dict[str, Any] = results) -> str:
            points = result_set[policy]
            values = []
            for index in (2, 4):
                point = points[index]
                values.append(f"{point['aggregate_tps']:.0f}/{point['per_user_tps']:.1f}")
            return " → ".join(values)

        lines.append(f"| {item['id']} | {cell('oracle-prefetch')} | {cell('fetch-at-decode')} |")
    return "\n".join(lines)


def tpot_summary(report: dict[str, Any]) -> dict[str, float]:
    """Small diagnostic used by tests and command-line smoke runs."""

    values = []
    for item in cast(Iterable[dict[str, Any]], report["series"]):
        point = cast(dict[str, Any], item["results"])["fp8"]["oracle-prefetch"][-1]
        values.append(float(point["tpot_ms"]))
    return {"mean_full_load_tpot_ms": statistics.fmean(values), "series": float(len(values))}
