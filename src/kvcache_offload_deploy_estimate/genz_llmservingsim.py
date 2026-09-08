"""GenZ roofline profiles bridged into LLMServingSim's profile-table contract.

The tracked GenZ release does not know the four frontier architectures in this
study.  This adapter reads their official Hugging Face configs, expresses the
active parameter path and long-context attention as aggregate GenZ operators,
and writes the resulting decode sweep as LLMServingSim ``per_sequence.csv``
bundles.  The production estimator reads those files through LLMServingSim's
own table builder and interpolation functions.

This is an analytical estimate, not a claim that the model kernels were run.
The A100 constants come from the two-GPU calibration artifact committed in this
repository; the 40% BF16 compute efficiency is the documented GenZ prior.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
GENZ_SOURCE = ROOT / "extern" / "tracked" / "genz-llm-analyzer"
if str(GENZ_SOURCE) not in sys.path:
    sys.path.insert(0, str(GENZ_SOURCE))

from GenZ.operator_base import Operator  # type: ignore[import-untyped]
from GenZ.system import System  # type: ignore[import-untyped]
from GenZ.unit import Unit  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class RooflineHardware:
    """Hardware inputs used by the GenZ operator model."""

    peak_bf16_tflops: float = 312.0
    compute_efficiency: float = 0.40
    hbm_stream_gbps: float = 1765.0
    # 51.6% of 2039 GB/s measured on the decode GEMM sweep, expressed
    # relative to the separately measured 1765 GB/s streaming ceiling.
    hbm_kernel_efficiency: float = (0.516 * 2039.0) / 1765.0
    fp8_dequant_gvalues_per_second: float = 455.0
    int4_dequant_gvalues_per_second: float = 455.0
    nvlink_latency_ms: float = 0.031
    nvlink_payload_gbps: float = 274.0


@dataclass(frozen=True, slots=True)
class FrontierModelSpec:
    """Model facts needed for decode and prefill operator construction."""

    key: str
    display_name: str
    config_submodule: str
    config_commit: str
    active_parameters_billions: float
    stored_weight_gb: float
    layers: int
    cache_layers: int
    hidden_size: int
    vocab_size: int
    experts: int
    experts_per_token: int
    moe_layers: int
    moe_intermediate_size: int
    qk_head_dim: int
    v_head_dim: int
    attention_heads: int
    index_heads: int
    index_dim: int
    index_topk: int
    full_index_layers: int
    linear_attention_layers: int
    index_pool: int
    tp_size: int
    pp_size: int
    nodes: int
    replicas: int
    nonexpert_weight_bits: int
    expert_weight_bits: int
    lm_head_weight_bits: int
    prefill_attention: str
    layer_types: tuple[str, ...]
    indexer_types: tuple[str, ...]
    compress_ratios: tuple[int, ...]
    sliding_window: int

    @property
    def active_parameters(self) -> float:
        return self.active_parameters_billions * 1e9

    @property
    def active_expert_parameters_per_token(self) -> float:
        return (
            self.moe_layers
            * self.experts_per_token
            * 3
            * self.hidden_size
            * self.moe_intermediate_size
        )

    @property
    def all_expert_parameters(self) -> float:
        return self.moe_layers * self.experts * 3 * self.hidden_size * self.moe_intermediate_size

    @property
    def lm_head_parameters(self) -> float:
        return self.hidden_size * self.vocab_size

    @property
    def nonexpert_parameters(self) -> float:
        return max(0.0, self.active_parameters - self.active_expert_parameters_per_token)


class _AggregateOperator(Operator):  # type: ignore[misc]
    """A shape-preserving aggregate accepted by GenZ's roofline implementation."""

    def __init__(
        self,
        name: str,
        *,
        macs: float,
        input_elements: float,
        stored_weight_bytes: float,
        output_elements: float,
    ) -> None:
        self.name = name
        self._macs = macs
        self._input_elements = input_elements
        # GenZ runs this operator with BF16 arithmetic.  Express arbitrary
        # stored precision as the equivalent number of two-byte elements.
        self._weight_elements = stored_weight_bytes / 2.0
        self._output_elements = output_elements
        super().__init__([macs, input_elements, stored_weight_bytes, output_elements, 3])

    def get_effective_dim_len(self) -> int:
        return 4

    def get_tensors(self) -> tuple[tuple[float], tuple[float], tuple[float]]:
        return (
            (self._input_elements,),
            (self._weight_elements,),
            (self._output_elements,),
        )

    def get_num_ops(self) -> float:
        # GenZ calls these MACs and multiplies by two for its FLOP reporting.
        return self._macs


PROFILE_ROOT = ROOT / "data" / "profiles" / "llmservingsim"
GENZ_COMMIT = "091bdd0a2777dfe4d405a8fcdd91c8f12b474bf3"
LLMSERVINGSIM_COMMIT = "a4053bc1161872420e1e0607cb3409ef659b828e"
CENTRAL_HARDWARE = RooflineHardware()


def _read_text_config(submodule: str) -> dict[str, Any]:
    data = json.loads((ROOT / "extern" / "tracked" / submodule / "config.json").read_text())
    return cast(dict[str, Any], data.get("text_config", data))


def _verified_spec(
    *,
    key: str,
    display_name: str,
    submodule: str,
    commit: str,
    active_b: float,
    stored_gb: float,
    cache_layers: int,
    tp: int,
    pp: int,
    nodes: int,
    replicas: int,
    nonexpert_bits: int,
    expert_bits: int,
    lm_head_bits: int,
    prefill_attention: str,
    full_index_layers: int = 0,
    linear_attention_layers: int = 0,
    index_pool: int = 1,
) -> FrontierModelSpec:
    config = _read_text_config(submodule)
    index_path = ROOT / "extern" / "tracked" / submodule / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        # Some source-only Hugging Face checkouts retain this file as a Git LFS
        # pointer. Their byte totals are pinned explicitly below.
        index = {}
    indexed_size = index.get("metadata", {}).get("total_size")
    if indexed_size is not None and not math.isclose(float(indexed_size), stored_gb * 1e9):
        raise ValueError(f"{display_name} stored-byte assumption disagrees with official index")
    if config.get("qk_head_dim") is not None:
        qk_raw = config["qk_head_dim"]
    elif config.get("qk_nope_head_dim") is not None:
        qk_raw = int(config["qk_nope_head_dim"]) + int(config.get("qk_rope_head_dim") or 0)
    else:
        # Some implementations expose only a full latent head dimension and a
        # RoPE sub-dimension. The latter is part of, not the entirety of, Q/K.
        qk_raw = config.get("head_dim")
    v_raw = config.get("v_head_dim") or config.get("head_dim")
    qk_dim = int(cast(str | int | float, qk_raw))
    v_dim = int(cast(str | int | float, v_raw))
    first_dense = int(config.get("first_k_dense_replace") or 0)
    layers = int(config["num_hidden_layers"])
    layer_types = cast(list[str], config.get("layer_types") or [])
    if layer_types:
        observed_linear = layer_types.count("linear_attention")
        if len(layer_types) != layers or observed_linear != linear_attention_layers:
            raise ValueError(f"{display_name} layer-type assumptions disagree with official config")
        if cache_layers != layers - observed_linear:
            raise ValueError(
                f"{display_name} cache-layer assumption disagrees with official config"
            )
    indexer_types = tuple(cast(list[str], config.get("indexer_types") or []))
    compress_ratios = tuple(int(value) for value in config.get("compress_ratios") or [])
    if indexer_types and len(indexer_types) < layers:
        raise ValueError(f"{display_name} indexer-type list is shorter than its layer count")
    if compress_ratios and len(compress_ratios) < layers:
        raise ValueError(f"{display_name} compression-ratio list is shorter than its layer count")
    return FrontierModelSpec(
        key=key,
        display_name=display_name,
        config_submodule=submodule,
        config_commit=commit,
        active_parameters_billions=active_b,
        stored_weight_gb=stored_gb,
        layers=layers,
        cache_layers=cache_layers,
        hidden_size=int(config["hidden_size"]),
        vocab_size=int(config["vocab_size"]),
        experts=int(config["n_routed_experts"]),
        experts_per_token=int(config["num_experts_per_tok"]),
        moe_layers=layers - first_dense,
        moe_intermediate_size=int(config["moe_intermediate_size"]),
        qk_head_dim=qk_dim,
        v_head_dim=v_dim,
        attention_heads=int(config["num_attention_heads"]),
        index_heads=int(config.get("index_n_heads") or 0),
        index_dim=int(config.get("index_head_dim") or 0),
        index_topk=int(config.get("index_topk") or 0),
        full_index_layers=full_index_layers,
        linear_attention_layers=linear_attention_layers,
        index_pool=index_pool,
        tp_size=tp,
        pp_size=pp,
        nodes=nodes,
        replicas=replicas,
        nonexpert_weight_bits=nonexpert_bits,
        expert_weight_bits=expert_bits,
        lm_head_weight_bits=lm_head_bits,
        prefill_attention=prefill_attention,
        layer_types=tuple(layer_types),
        indexer_types=indexer_types,
        compress_ratios=compress_ratios[:layers],
        sliding_window=int(config.get("sliding_window") or 0),
    )


MODEL_SPECS: dict[str, FrontierModelSpec] = {
    "kimi": _verified_spec(
        key="kimi",
        display_name="Kimi Code 2.7",
        submodule="kimi-k2.7-code",
        commit="74797c9c62378b951a1f6fcf5c4631024e9b8bef",
        active_b=32.0,
        stored_gb=37.2 * 16,
        cache_layers=61,
        tp=2,
        pp=8,
        nodes=2,
        replicas=1,
        nonexpert_bits=16,
        expert_bits=4,
        lm_head_bits=16,
        prefill_attention="dense-mla",
    ),
    "glm": _verified_spec(
        key="glm",
        display_name="GLM-5.3",
        submodule="glm-5.3",
        commit="aca966e4e02791568aa6a4ced368624b3d897f42",
        active_b=40.0,
        stored_gb=47.225 * 16,
        cache_layers=78,
        tp=2,
        pp=8,
        nodes=2,
        replicas=1,
        nonexpert_bits=8,
        expert_bits=8,
        lm_head_bits=16,
        prefill_attention="dsa",
        full_index_layers=21,
    ),
    "glm-flash": _verified_spec(
        key="glm-flash",
        display_name="GLM-5.3-Flash",
        submodule="glm-5.3-flash",
        commit="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        active_b=18.0,
        stored_gb=328.326771576,
        cache_layers=11,
        tp=8,
        pp=1,
        nodes=1,
        replicas=1,
        # The official FP8 checkpoint excludes the linear-attention stack,
        # sparse indexer/latent projections, embeddings, and LM head.
        nonexpert_bits=16,
        expert_bits=8,
        lm_head_bits=16,
        prefill_attention="hybrid-kda-dsa",
        full_index_layers=11,
        linear_attention_layers=34,
        index_pool=4,
    ),
    "deepseek-flash": _verified_spec(
        key="deepseek-flash",
        display_name="DeepSeek V4 Flash",
        submodule="deepseek-v4-flash",
        commit="60d8d70770c6776ff598c94bb586a859a38244f1",
        active_b=13.0,
        stored_gb=159.609485896,
        cache_layers=21,
        tp=4,
        pp=1,
        nodes=1,
        replicas=2,
        nonexpert_bits=8,
        expert_bits=4,
        lm_head_bits=8,
        prefill_attention="compressed-sparse",
        full_index_layers=40,
    ),
}


def _system(hardware: RooflineHardware) -> System:
    # GenZ's ``GBsec`` unit is binary (2**30 bytes/s), whereas the measured
    # bandwidths in this project use decimal GB/s. Normalize before passing
    # the value so the roofline sees the intended byte rate.
    genz_hbm_gibps = hardware.hbm_stream_gbps * 1e9 / 2**30
    return System(
        flops=hardware.peak_bf16_tflops,
        offchip_mem_bw=genz_hbm_gibps,
        bits="bf16",
        compute_efficiency=hardware.compute_efficiency,
        memory_efficiency=hardware.hbm_kernel_efficiency,
        comm_efficiency=1.0,
    )


def _operator_ms(
    name: str,
    *,
    macs: float,
    weight_values: float,
    weight_bits: int,
    activation_elements: float,
    hardware: RooflineHardware,
) -> float:
    stored_weight_bytes = weight_values * weight_bits / 8.0
    operator = _AggregateOperator(
        name,
        macs=macs,
        input_elements=activation_elements / 2.0,
        stored_weight_bytes=stored_weight_bytes,
        output_elements=activation_elements / 2.0,
    )
    row = operator.get_roofline(_system(hardware), Unit())
    roofline_ms = float(row["Latency (msec)"])
    if weight_bits == 8:
        dequant_ms = weight_values / (hardware.fp8_dequant_gvalues_per_second * 1e9) * 1000.0
    elif weight_bits == 4:
        dequant_ms = weight_values / (hardware.int4_dequant_gvalues_per_second * 1e9) * 1000.0
    else:
        dequant_ms = 0.0
    # Successful fused conversion: the conversion, HBM read, and BF16 math
    # share a roofline.  An unfused implementation would sum these terms.
    return max(roofline_ms, dequant_ms)


def expected_distinct_experts(spec: FrontierModelSpec, sequences: int) -> float:
    """Uniform-routing occupancy expectation for one MoE layer."""

    draws = sequences * spec.experts_per_token
    return spec.experts * (1.0 - (1.0 - 1.0 / spec.experts) ** draws)


def _tp_collective_ms(spec: FrontierModelSpec, sequences: int, hardware: RooflineHardware) -> float:
    if spec.tp_size <= 1:
        return 0.0
    payload = sequences * spec.hidden_size * 2.0
    one_allreduce = (
        2.0 * (spec.tp_size - 1) * hardware.nvlink_latency_ms
        + 2.0
        * (spec.tp_size - 1)
        / spec.tp_size
        * payload
        / (hardware.nvlink_payload_gbps * 1e9)
        * 1000.0
    )
    return 2.0 * spec.layers * one_allreduce


@lru_cache(maxsize=4096)
def _decode_cached(
    model_key: str,
    sequences: int,
    hardware: RooflineHardware,
) -> tuple[float, dict[str, float]]:
    spec = MODEL_SPECS[model_key]
    tp = spec.tp_size
    active_expert = spec.active_expert_parameters_per_token
    distinct = expected_distinct_experts(spec, sequences)
    streamed_expert = spec.all_expert_parameters * distinct / spec.experts

    lm_head = min(spec.lm_head_parameters, spec.nonexpert_parameters)
    shared = spec.nonexpert_parameters - lm_head
    activation_elements = sequences * spec.hidden_size * spec.layers * 4.0 / tp
    shared_ms = _operator_ms(
        "shared_path",
        macs=shared * sequences / tp,
        weight_values=shared / tp,
        weight_bits=spec.nonexpert_weight_bits,
        activation_elements=activation_elements,
        hardware=hardware,
    )
    lm_head_ms = _operator_ms(
        "lm_head",
        macs=lm_head * sequences / tp,
        weight_values=lm_head / tp,
        weight_bits=spec.lm_head_weight_bits,
        activation_elements=sequences * (spec.hidden_size + spec.vocab_size) / tp,
        hardware=hardware,
    )
    expert_ms = _operator_ms(
        "routed_experts",
        macs=active_expert * sequences / tp,
        weight_values=streamed_expert / tp,
        weight_bits=spec.expert_weight_bits,
        activation_elements=activation_elements,
        hardware=hardware,
    )
    collective_ms = _tp_collective_ms(spec, sequences, hardware)
    components = {
        "shared_ms": shared_ms,
        "lm_head_ms": lm_head_ms,
        "expert_ms": expert_ms,
        "tp_collective_ms": collective_ms,
        "expected_distinct_experts": distinct,
    }
    return sum(value for key, value in components.items() if key.endswith("_ms")), components


def analytical_decode_ms(
    model_key: str,
    sequences: int,
    hardware: RooflineHardware = CENTRAL_HARDWARE,
) -> float:
    if sequences < 1:
        raise ValueError("sequences must be positive")
    return _decode_cached(model_key, sequences, hardware)[0]


def _stage_layer_counts(layers: int, pp_size: int) -> tuple[int, ...]:
    """Mirror LLMServingSim/vLLM's block-balanced PP partition rule."""

    per_stage = layers // pp_size
    partitions = [per_stage] * pp_size
    for index in range(2, layers % pp_size + 2):
        partitions[-index] += 1
    return tuple(partitions)


def analytical_decode_stage_ms(
    model_key: str,
    sequences: int,
    hardware: RooflineHardware = CENTRAL_HARDWARE,
) -> tuple[float, ...]:
    """Split the GenZ core time over balanced PP stages.

    Shared and routed transformer work follows block counts. The vocabulary
    projection remains on the last stage, matching LLMServingSim's PP trace.
    """

    spec = MODEL_SPECS[model_key]
    breakdown = decode_breakdown(model_key, sequences, hardware)
    distributed_ms = breakdown["shared_ms"] + breakdown["expert_ms"] + breakdown["tp_collective_ms"]
    counts = _stage_layer_counts(spec.layers, spec.pp_size)
    stages = [distributed_ms * count / spec.layers for count in counts]
    stages[-1] += breakdown["lm_head_ms"]
    return tuple(stages)


def decode_breakdown(
    model_key: str,
    sequences: int,
    hardware: RooflineHardware = CENTRAL_HARDWARE,
) -> dict[str, float]:
    return dict(_decode_cached(model_key, sequences, hardware)[1])


def _selected_attention_flops(spec: FrontierModelSpec, tokens: int, layers: int) -> float:
    selected = min(float(spec.index_topk or tokens), (tokens + 1.0) / 2.0)
    return (
        2.0
        * tokens
        * selected
        * spec.attention_heads
        * (spec.qk_head_dim + spec.v_head_dim)
        * layers
    )


def _prefill_attention_flops(spec: FrontierModelSpec, tokens: int) -> float:
    if spec.prefill_attention == "dense-mla":
        return tokens**2 * spec.attention_heads * (spec.qk_head_dim + spec.v_head_dim) * spec.layers
    if spec.prefill_attention == "dsa":
        index = tokens**2 * spec.index_heads * spec.index_dim * spec.full_index_layers
        return index + _selected_attention_flops(spec, tokens, spec.cache_layers)
    if spec.prefill_attention == "hybrid-kda-dsa":
        pooled_index = (
            tokens**2 / spec.index_pool * spec.index_heads * spec.index_dim * spec.full_index_layers
        )
        sparse = _selected_attention_flops(spec, tokens, spec.cache_layers)
        kda = (
            6.0 * tokens * spec.attention_heads * spec.qk_head_dim**2 * spec.linear_attention_layers
        )
        return pooled_index + sparse + kda
    if spec.prefill_attention == "compressed-sparse":
        # Official V4-Flash ratios: 21 layers at 4x compression and 20 at
        # 128x, after two zero/special entries. Hash layers are conservatively
        # represented by the same compressed-score arithmetic.
        compressed_index = (
            tokens**2 * spec.index_heads * spec.index_dim * (21.0 / 4.0 + 20.0 / 128.0)
        )
        return compressed_index + _selected_attention_flops(spec, tokens, spec.cache_layers)
    raise ValueError(f"unknown prefill attention model: {spec.prefill_attention}")


@lru_cache(maxsize=256)
def analytical_prefill_seconds(
    model_key: str,
    tokens: int,
    hardware: RooflineHardware = CENTRAL_HARDWARE,
) -> float:
    """Chunked-prefill roofline for one text-only request."""

    if tokens < 1:
        raise ValueError("tokens must be positive")
    spec = MODEL_SPECS[model_key]
    devices = spec.tp_size * spec.pp_size
    chunks = math.ceil(tokens / 2048)
    linear_flops = 2.0 * spec.active_parameters * tokens
    attention_flops = _prefill_attention_flops(spec, tokens)
    total_macs_per_device = (linear_flops + attention_flops) / 2.0 / devices
    # Official checkpoint bytes are streamed approximately once per prefill
    # chunk.  This captures the loss of weight reuse across chunk boundaries.
    stored_bytes_per_device = spec.stored_weight_gb * 1e9 * chunks / devices
    activation_elements = tokens * spec.hidden_size * spec.layers * 8.0 / devices
    op = _AggregateOperator(
        "chunked_prefill",
        macs=total_macs_per_device,
        input_elements=activation_elements / 2.0,
        stored_weight_bytes=stored_bytes_per_device,
        output_elements=activation_elements / 2.0,
    )
    roofline_ms = float(op.get_roofline(_system(hardware), Unit())["Latency (msec)"])
    fill = (chunks + spec.pp_size - 1) / chunks
    return roofline_ms * fill / 1000.0


def _llmservingsim_trace_module() -> Any:
    root = ROOT / "extern" / "tracked" / "llmservingsim"
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module("serving.core.trace_generator")


def profile_variant_root(model_key: str) -> Path:
    return PROFILE_ROOT / "A100-SXM4-80GB" / model_key / "official-dequant-bf16"


def generate_profile_bundles() -> list[Path]:
    """Write deterministic GenZ results using LLMServingSim's bundle schema."""

    written: list[Path] = []
    for spec in MODEL_SPECS.values():
        variant_root = profile_variant_root(spec.key)
        tp_root = variant_root / f"tp{spec.tp_size}"
        tp_root.mkdir(parents=True, exist_ok=True)
        csv_path = tp_root / "per_sequence.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("layer", "sequences", "time_us"))
            # Full-resident no-offload baselines can admit several hundred
            # compressed-cache requests. Keep those local-batch rows inside
            # LLMServingSim's interpolation domain instead of clamping at 128.
            # The corrected HBM-only ShadowKV boundary reaches 226 sequences for
            # the smallest-cache case, while native compressed-cache controls and
            # MTP verification can still expand the queried local batch. Keep all
            # displayed points comfortably inside the table.
            for sequences in range(1, 4097):
                total_ms = analytical_decode_ms(spec.key, sequences)
                writer.writerow(
                    (
                        "model_core",
                        sequences,
                        f"{total_ms * 1000.0:.9f}",
                    )
                )
                for stage, stage_ms in enumerate(analytical_decode_stage_ms(spec.key, sequences)):
                    writer.writerow(
                        (f"model_core_stage_{stage}", sequences, f"{stage_ms * 1000.0:.9f}")
                    )
        meta = {
            "profiler_version": "GenZ-bridge-1",
            "hardware": "A100-SXM4-80GB",
            "model": spec.display_name,
            "variant": "official-dequant-bf16",
            "tp_degrees": [spec.tp_size],
            "source": {
                "genz_commit": GENZ_COMMIT,
                "llmservingsim_commit": LLMSERVINGSIM_COMMIT,
                "config_submodule": spec.config_submodule,
                "config_commit": spec.config_commit,
            },
            "hardware_inputs": asdict(CENTRAL_HARDWARE),
            "model_inputs": asdict(spec),
        }
        (variant_root / "meta.yaml").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.extend((csv_path, variant_root / "meta.yaml"))
    return written


@lru_cache(maxsize=32)
def _profile_db(model_key: str) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    root = profile_variant_root(model_key)
    if not (root / f"tp{spec.tp_size}" / "per_sequence.csv").exists():
        generate_profile_bundles()
    names = ["model_core", *(f"model_core_stage_{stage}" for stage in range(spec.pp_size))]
    return {
        "meta": {},
        "architecture": {
            "catalog": {"per_sequence": {name: {"tp_stable": False} for name in names}},
            "sequence": {},
        },
        "variant": "official-dequant-bf16",
        "hardware": "A100-SXM4-80GB",
        "model": spec.display_name,
        "root": str(root),
        "available_tps": [spec.tp_size],
        "tables": {},
    }


def llmservingsim_decode_ms(model_key: str, sequences: int) -> float:
    """Read a GenZ row through LLMServingSim's production interpolation path."""

    spec = MODEL_SPECS[model_key]
    module = _llmservingsim_trace_module()
    latency_ns = module._lookup_per_sequence(
        _profile_db(model_key), "model_core", spec.tp_size, sequences
    )
    return float(latency_ns) / 1e6


def llmservingsim_decode_stage_ms(model_key: str, sequences: int) -> tuple[float, ...]:
    """Read every generated PP-stage row through LLMServingSim interpolation."""

    spec = MODEL_SPECS[model_key]
    module = _llmservingsim_trace_module()
    return tuple(
        float(
            module._lookup_per_sequence(
                _profile_db(model_key), f"model_core_stage_{stage}", spec.tp_size, sequences
            )
        )
        / 1e6
        for stage in range(spec.pp_size)
    )


def profile_manifest() -> dict[str, Any]:
    return {
        "genz_commit": GENZ_COMMIT,
        "llmservingsim_commit": LLMSERVINGSIM_COMMIT,
        "profile_root": str(PROFILE_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "hardware": asdict(CENTRAL_HARDWARE),
        "models": {key: asdict(value) for key, value in MODEL_SPECS.items()},
    }
