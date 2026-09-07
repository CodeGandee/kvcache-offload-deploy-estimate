"""ShadowKV trace extension for LLMServingSim-style decode studies.

The upstream simulator models ordinary tiered KV blocks.  ShadowKV has a different
per-transformer-block dependency: landmark selection must finish before cache misses
are known, key reconstruction and value fetch overlap, and sparse attention waits for
both.  This module adds those events without modifying the tracked upstream source.

The non-ShadowKV model-forward component is a calibration input.  Everything added by
this module (selection, prefetch traffic, miss materialization, and FP8 conversion) is
calculated explicitly and is therefore independently inspectable.
"""

from __future__ import annotations

import importlib
import math
import random
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

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
    """Usable, not nameplate, bandwidth assumptions for one 8×A800 server."""

    pcie_per_gpu_gbps: float = 25.0
    host_dram_gbps: float = 180.0
    fp8_dequant_gbps: float = 360.0
    nvlink_latency_ms: float = 0.006
    ib_latency_ms: float = 0.012
    ib_payload_gbps: float = 40.0


@dataclass(frozen=True, slots=True)
class Scenario:
    """One model/context placement and its calibrated non-ShadowKV decode profile."""

    id: str
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
    reference_bf16_tpot_ms: tuple[float, float, float, float, float]
    ttft_seconds: tuple[float, float, float, float, float]
    last_ttft_seconds: tuple[float, float, float, float, float]
    weight_per_gpu_gb: float
    mtp_draft_tokens: int = 0

    @property
    def load_users(self) -> tuple[int, int, int, int, int]:
        def half_up(value: float) -> int:
            return math.floor(value + 0.5)

        loads = tuple(max(1, half_up(self.max_users * load)) for load in (0.25, 0.5, 0.75, 1.0))
        return 1, loads[0], loads[1], loads[2], loads[3]

    def reference_tpot_for_users(self, users: int) -> float:
        """Linearly interpolate the calibrated core profile at an arbitrary load."""

        if users < 1 or users > self.max_users:
            raise ValueError("users must be between one and the admission ceiling")
        points = tuple(zip(self.load_users, self.reference_bf16_tpot_ms, strict=True))
        for point_users, value in points:
            if users == point_users:
                return value
        for (left_users, left), (right_users, right) in pairwise(points):
            if left_users <= users <= right_users:
                fraction = (users - left_users) / (right_users - left_users)
                return left + fraction * (right - left)
        raise AssertionError("reference interpolation did not bracket the requested load")


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


def _mtp_core_position_fraction(scenario: Scenario, users: int) -> float:
    """Fraction of the calibrated non-ShadowKV path that scales per verified position.

    The remainder represents shared weight reads, pipeline launch/bubble time, and
    communication.  This is an explicit calibration assumption, not an upstream
    LLMServingSim result.
    """

    utilization = users / scenario.max_users
    if scenario.pp_size > 1:
        return 0.15 + 0.10 * utilization
    return 0.30 + 0.25 * utilization


def _transfer_time_ms(
    *,
    stage_bytes_per_gpu: float,
    factor: float,
    users: int,
    tp_size: int,
    hardware: A800Host,
) -> float:
    per_gpu = stage_bytes_per_gpu * factor * users / (hardware.pcie_per_gpu_gbps * 1e9)
    host = stage_bytes_per_gpu * tp_size * factor * users / (hardware.host_dram_gbps * 1e9)
    return max(per_gpu, host) * 1000.0


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
    layer_counts = stage_layer_counts(scenario.cache_layers, scenario.pp_size)
    resident_counts = _resident_counts_by_stage(layer_counts, resident_layers)
    stage_bytes = _stage_cache_bytes(scenario, storage=storage)

    # Remove the earlier report's simple 80/80 JIT-transfer allowance to obtain
    # the non-ShadowKV profile floor, then insert the explicit block events below.
    reference_oracle = OraclePrefetch()
    _, reference_jit = reference_oracle.traffic_factors(reuse=reuse)
    reference_jit_ms = sum(
        _transfer_time_ms(
            stage_bytes_per_gpu=value * (2.0 / stored_bytes_per_value(storage)),
            factor=reference_jit,
            users=local_users,
            tp_size=scenario.tp_size,
            hardware=hardware,
        )
        for value in stage_bytes
    )
    reference_tpot = (
        scenario.reference_bf16_tpot_ms[load_index]
        if users_override is None
        else scenario.reference_tpot_for_users(users)
    )
    profile_floor = max(0.1, reference_tpot - reference_jit_ms)
    core_scale = 1.0 + _mtp_core_position_fraction(scenario, users) * (verification_tokens - 1)

    events: list[TraceEvent] = []
    selector_block = selector_ms_per_block(local_users, scenario.context_tokens) * selector_scale
    if policy == "oracle-prefetch":
        prefetch_factor, jit_factor = oracle.traffic_factors(reuse=reuse)
        miss_fraction = jit_factor
    else:
        prefetch_factor = 0.0
        miss_fraction = 1.0 - reuse

    materialize_block, _, _ = _paper_materialize_ms_per_block(
        users=local_users,
        context_tokens=scenario.context_tokens,
        cached_width=scenario.cached_width,
        miss_fraction=miss_fraction,
        storage=storage,
        dequant_gbps=hardware.fp8_dequant_gbps,
    )
    materialize_block *= materialize_scale
    resident_materialize_block = (
        _resident_materialize_ms_per_block(
            users=local_users,
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
        extra_users = local_users * (verification_tokens - 1)
        extra_selector_block = (
            selector_ms_per_block(local_users * verification_tokens, scenario.context_tokens)
            - selector_ms_per_block(local_users, scenario.context_tokens)
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

    critical_added = 0.0
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
            users=local_users,
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
            users=local_users,
            tp_size=scenario.tp_size,
            hardware=hardware,
        )
        extra_materialize = max(extra_materialize, extra_transfer)
        stage_critical = selection + materialize + extra_selection + extra_materialize
        critical_added += stage_critical
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
                    * local_users
                ),
            )
        )

    background_stall = 0.0
    if prefetch_factor > 0.0:
        node_bytes = [0.0] * scenario.nodes
        per_gpu_times: list[float] = []
        for stage, (layers, resident, bytes_for_stage) in enumerate(
            zip(layer_counts, resident_counts, stage_bytes, strict=True)
        ):
            node = min(scenario.nodes - 1, stage * scenario.nodes // scenario.pp_size)
            moved = bytes_for_stage * (layers - resident) / layers * prefetch_factor * local_users
            node_bytes[node] += moved * scenario.tp_size
            per_gpu_times.append(moved / (hardware.pcie_per_gpu_gbps * 1e9) * 1000.0)
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
            max(
                (value / (hardware.host_dram_gbps * 1e9) * 1000.0 for value in node_bytes),
                default=0.0,
            ),
        )
        # One-token lookahead has the preceding token's non-ShadowKV forward time
        # as its overlap window.  More lookahead scales that window linearly.
        overlap_window = profile_floor * oracle.lookahead_tokens
        background_stall = max(0.0, background_service - overlap_window)

    communication = 0.0
    if scenario.pp_size > 1:
        communication = (scenario.pp_size - scenario.nodes) * hardware.nvlink_latency_ms
        communication += (scenario.nodes - 1) * hardware.ib_latency_ms
    mtp_draft = 0.0
    if verification_tokens > 1:
        draft_core = profile_floor * verification_tokens * 1.5 / scenario.model_layers
        draft_materialize, _, _ = _paper_materialize_ms_per_block(
            users=local_users,
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

    round_total = (
        profile_floor * core_scale + critical_added + background_stall + communication + mtp_draft
    )
    return round_total / emitted_tokens, tuple(events)


DEFAULT_ORACLE = OraclePrefetch()
DEFAULT_HARDWARE = A800Host()


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
            pcie_per_gpu_gbps=rng.triangular(18.0, 29.0, hardware.pcie_per_gpu_gbps),
            host_dram_gbps=rng.triangular(130.0, 220.0, hardware.host_dram_gbps),
            fp8_dequant_gbps=rng.triangular(240.0, 520.0, hardware.fp8_dequant_gbps),
            nvlink_latency_ms=hardware.nvlink_latency_ms,
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
    return PointEstimate(
        users=users,
        users_per_replica=math.ceil(users / scenario.replicas),
        tpot_ms=central,
        aggregate_tps=aggregate,
        per_user_tps=1000.0 / central,
        sensitivity_p10_ms=percentile(samples, 0.10),
        sensitivity_p90_ms=percentile(samples, 0.90),
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
    Scenario(
        "kimi-128",
        "Kimi Code 2.7",
        131_072,
        28,
        61,
        61,
        576,
        8,
        2,
        2,
        1,
        (166.7, 73.7, 66.4, 71.2, 79.3),
        (16.6, 66.4, 124.5, 182.6, 240.7),
        (16.6, 116.2, 232.4, 348.6, 464.8),
        37.2,
        0,
    ),
    Scenario(
        "kimi-256",
        "Kimi Code 2.7",
        262_144,
        14,
        61,
        61,
        576,
        8,
        2,
        2,
        1,
        (200.0, 133.3, 90.9, 85.3, 81.9),
        (43.2, 108.0, 172.8, 259.2, 324.0),
        (43.2, 172.8, 302.4, 475.2, 604.8),
        37.2,
        0,
    ),
    Scenario(
        "glm-128",
        "GLM-5.3",
        131_072,
        40,
        78,
        78,
        576,
        8,
        2,
        2,
        1,
        (333.3, 129.9, 127.4, 138.9, 153.8),
        (14.3, 78.7, 150.2, 221.7, 293.2),
        (14.3, 143.0, 286.0, 429.0, 572.0),
        47.225,
        3,
    ),
    Scenario(
        "glm-256",
        "GLM-5.3",
        262_144,
        30,
        78,
        78,
        576,
        8,
        2,
        2,
        1,
        (500.0, 142.9, 123.9, 130.7, 142.2),
        (28.7, 129.2, 229.6, 344.4, 444.9),
        (28.7, 229.6, 430.5, 660.1, 861.0),
        47.225,
        3,
    ),
    Scenario(
        "flash-128",
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
        (24.4, 56.7, 79.4, 101.0, 124.5),
        (7.3, 32.8, 61.9, 91.0, 120.1),
        (7.3, 58.2, 116.5, 174.7, 232.9),
        39.9,
        2,
    ),
    Scenario(
        "flash-256",
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
        (27.8, 53.1, 71.4, 89.4, 109.9),
        (14.6, 51.0, 94.6, 138.3, 182.0),
        (14.6, 87.4, 174.7, 262.1, 349.4),
        39.9,
        2,
    ),
    Scenario(
        "kimi-72",
        "Kimi Code 2.7",
        73_728,
        48,
        61,
        61,
        576,
        8,
        2,
        2,
        1,
        (150.0, 61.0, 56.0, 59.0, 65.0),
        cast(tuple[float, float, float, float, float], _burst_ttft(7.5, 48, 1)[0]),
        cast(tuple[float, float, float, float, float], _burst_ttft(7.5, 48, 1)[1]),
        37.2,
        0,
    ),
    Scenario(
        "glm-72",
        "GLM-5.3",
        73_728,
        48,
        78,
        78,
        576,
        8,
        2,
        2,
        1,
        (300.0, 115.0, 112.0, 122.0, 137.0),
        cast(tuple[float, float, float, float, float], _burst_ttft(8.0, 48, 1)[0]),
        cast(tuple[float, float, float, float, float], _burst_ttft(8.0, 48, 1)[1]),
        47.225,
        3,
    ),
    Scenario(
        "flash-72",
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
        (21.0, 50.0, 72.0, 94.0, 116.0),
        cast(tuple[float, float, float, float, float], _burst_ttft(4.1, 80, 2)[0]),
        cast(tuple[float, float, float, float, float], _burst_ttft(4.1, 80, 2)[1]),
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
            cast(dict[str, Any], item["results"])[storage] = storage_results
        series.append(item)
    return {
        "schema_version": 1,
        "engine": {
            "name": "LLMServingSim 2.0 + external ShadowKV trace extension",
            "upstream_commit": "a4053bc1161872420e1e0607cb3409ef659b828e",
            "integration": "upstream _pp_stage_boundaries plus added per-block trace events",
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
        "series": series,
    }


def _compact_point(point: PointEstimate) -> dict[str, float | int]:
    return {
        "users": point.users,
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
            "base": {},
            "mtp": {},
            "residency": {},
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
            cast(dict[str, Any], item["base"])[storage] = base_storage
            cast(dict[str, Any], item["mtp"])[storage] = mtp_storage
            cast(dict[str, Any], item["residency"])[storage] = residency_storage
        series.append(item)
    return {
        "schemaVersion": 2,
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
