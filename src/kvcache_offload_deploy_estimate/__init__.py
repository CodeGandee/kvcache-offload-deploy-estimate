"""Deployment-estimation primitives for KV-cache offloading studies."""

from .model import (
    PP2_TP8,
    PP8_TP2,
    ParallelTopology,
    average_burst_ttft,
    pipeline_efficiency,
    pp8_to_pp2_throughput_ratio,
    selected_entries,
    selected_payload_bytes_per_gpu,
)

__all__ = [
    "PP2_TP8",
    "PP8_TP2",
    "ParallelTopology",
    "average_burst_ttft",
    "pipeline_efficiency",
    "pp8_to_pp2_throughput_ratio",
    "selected_entries",
    "selected_payload_bytes_per_gpu",
]

__version__ = "0.1.0"
