"""Small, auditable formulas used by the deployment estimates."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True, slots=True)
class ParallelTopology:
    """A pipeline/tensor-parallel layout spread across one or more nodes."""

    pipeline_stages: int
    tensor_parallel: int
    nodes: int
    stages_per_node: int

    def __post_init__(self) -> None:
        values = (
            self.pipeline_stages,
            self.tensor_parallel,
            self.nodes,
            self.stages_per_node,
        )
        if any(value <= 0 for value in values):
            raise ValueError("topology values must be positive")
        if self.nodes * self.stages_per_node != self.pipeline_stages:
            raise ValueError("nodes × stages_per_node must equal pipeline_stages")

    @property
    def total_gpus(self) -> int:
        """Return the total number of GPUs occupied by the deployment."""

        return self.pipeline_stages * self.tensor_parallel

    @property
    def average_layer_share_per_gpu(self) -> float:
        """Return the balanced fraction of model layers owned by one GPU."""

        return 1.0 / self.total_gpus

    @property
    def average_layer_share_per_node(self) -> float:
        """Return the balanced fraction of model layers owned by one node."""

        return self.stages_per_node / self.pipeline_stages


PP8_TP2 = ParallelTopology(
    pipeline_stages=8,
    tensor_parallel=2,
    nodes=2,
    stages_per_node=4,
)
PP2_TP8 = ParallelTopology(
    pipeline_stages=2,
    tensor_parallel=8,
    nodes=2,
    stages_per_node=1,
)


def pipeline_efficiency(pipeline_stages: int, microbatches: int) -> float:
    """Return ideal forward-only pipeline fill efficiency.

    The model assumes balanced stages and a makespan of ``m + p - 1`` stage slots.
    """

    if pipeline_stages <= 0 or microbatches <= 0:
        raise ValueError("pipeline_stages and microbatches must be positive")
    return microbatches / (microbatches + pipeline_stages - 1)


def selected_entries(context_tokens: int, fraction: float = 1.0 / 64.0) -> int:
    """Return the number of context entries admitted by the sparse selector."""

    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return ceil(context_tokens * fraction)


def selected_payload_bytes_per_gpu(
    *,
    context_tokens: int,
    cache_layers: int,
    cached_width: int,
    bytes_per_value: float,
    topology: ParallelTopology,
    fraction: float = 1.0 / 64.0,
) -> float:
    """Return balanced selected-cache bytes per GPU and request.

    The formula assumes that pipeline stages own disjoint layer ranges and TP ranks
    shard the selected cache rather than replicate it.
    """

    if cache_layers <= 0 or cached_width <= 0 or bytes_per_value <= 0:
        raise ValueError("cache dimensions and bytes_per_value must be positive")
    return (
        selected_entries(context_tokens, fraction)
        * cache_layers
        * cached_width
        * bytes_per_value
        / topology.total_gpus
    )


def pp8_to_pp2_throughput_ratio(
    microbatches: int,
    *,
    tp2_collective_credit: float = 1.03,
    single_user_ratio: float = 0.24,
) -> float:
    """Return the report's PP8×TP2/PP2×TP8 decode-throughput ratio.

    Both layouts occupy 16 GPUs and have the same asymptotic two-node capacity.
    The ratio therefore comes from pipeline bubbles, with a small TP2 collective
    credit. The one-user point includes the extra stage-handoff penalty used in
    the report.
    """

    if microbatches <= 0:
        raise ValueError("microbatches must be positive")
    if tp2_collective_credit <= 0 or single_user_ratio <= 0:
        raise ValueError("calibration factors must be positive")
    if microbatches == 1:
        return single_user_ratio
    return tp2_collective_credit * (microbatches + 1) / (microbatches + 7)


def prefill_pp8_to_pp2_ratio(
    prompt_chunks: int,
    *,
    tp2_collective_credit: float = 1.03,
) -> float:
    """Return the PP8×TP2/PP2×TP8 prefill-throughput ratio."""

    if prompt_chunks <= 0 or tp2_collective_credit <= 0:
        raise ValueError("prompt_chunks and calibration factor must be positive")
    return tp2_collective_credit * (prompt_chunks + 1) / (prompt_chunks + 7)


def average_burst_ttft(
    single_prompt_ttft_seconds: float,
    requests: int,
    *,
    replicas: int = 1,
) -> float:
    """Estimate mean first-token latency for a simultaneous closed burst."""

    if single_prompt_ttft_seconds <= 0 or requests <= 0 or replicas <= 0:
        raise ValueError("TTFT, requests, and replicas must be positive")
    requests_per_replica = ceil(requests / replicas)
    return single_prompt_ttft_seconds * (requests_per_replica + 1) / 2


def last_user_ttft(
    single_prompt_ttft_seconds: float,
    requests: int,
    *,
    replicas: int = 1,
) -> float:
    """Estimate last-user first-token latency for the same closed burst."""

    if single_prompt_ttft_seconds <= 0 or requests <= 0 or replicas <= 0:
        raise ValueError("TTFT, requests, and replicas must be positive")
    return single_prompt_ttft_seconds * ceil(requests / replicas)
