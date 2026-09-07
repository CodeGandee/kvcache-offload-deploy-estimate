"""Synthetic calibration of A100 decode and KV-offload primitives."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

GIB = 1024**3
A100_SXM4_80GB_PEAK_GBPS = 2039.0
A100_BF16_PEAK_TFLOPS = 312.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def iterations_for(nbytes: int, target_gib: float = 8.0) -> int:
    return max(3, min(200, round(target_gib * GIB / nbytes)))


def cuda_event_ms(
    fn: Callable[[], None],
    *,
    device: int,
    iterations: int,
    warmup: int = 4,
    rounds: int = 7,
) -> list[float]:
    with torch.cuda.device(device):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        result: list[float] = []
        for _ in range(rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            end.synchronize()
            result.append(start.elapsed_time(end) / iterations)
        return result


def run_text(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def device_metadata(device: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "logical_device": device,
        "name": props.name,
        "total_memory_gib": props.total_memory / GIB,
        "multi_processor_count": props.multi_processor_count,
        "major": props.major,
        "minor": props.minor,
    }


def hbm_bandwidth(device: int, size_gib: float = 1.0) -> dict[str, Any]:
    count = int(size_gib * GIB) // 4
    with torch.cuda.device(device):
        src = torch.empty(count, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)
        src.fill_(1.0)
        dst.zero_()
        nbytes = src.numel() * src.element_size()
        iterations = iterations_for(nbytes)

        copy_ms = cuda_event_ms(lambda: dst.copy_(src), device=device, iterations=iterations)
        add_ms = cuda_event_ms(
            lambda: torch.add(src, 1.0, out=dst),
            device=device,
            iterations=iterations,
        )

    def convert(times: list[float]) -> dict[str, Any]:
        rates = [2 * nbytes / (value / 1000.0) / 1e9 for value in times]
        return {
            "latency_ms": summary(times),
            "gbps_read_plus_write": summary(rates),
            "fraction_of_2039_gbps": statistics.median(rates) / A100_SXM4_80GB_PEAK_GBPS,
        }

    return {
        "working_set_gib_each": nbytes / GIB,
        "iterations_per_round": iterations,
        "d2d_copy": convert(copy_ms),
        "sm_add_out": convert(add_ms),
    }


def h2d_single(device: int, sizes_mib: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for size_mib in sizes_mib:
        nbytes = round(size_mib * 1024**2)
        host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        host.fill_(17)
        with torch.cuda.device(device):
            dst = torch.empty(nbytes, dtype=torch.uint8, device=device)
            iterations = iterations_for(nbytes, target_gib=4.0)

            def copy_operation(
                source: torch.Tensor = host, destination: torch.Tensor = dst
            ) -> None:
                destination.copy_(source, non_blocking=True)

            times = cuda_event_ms(
                copy_operation,
                device=device,
                iterations=iterations,
            )
        rates = [nbytes / (value / 1000.0) / 1e9 for value in times]
        rows.append(
            {
                "size_mib": size_mib,
                "iterations_per_round": iterations,
                "latency_ms": summary(times),
                "payload_gbps": summary(rates),
            }
        )
        del host, dst
    return rows


def h2d_dual(sizes_mib: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for size_mib in sizes_mib:
        nbytes = round(size_mib * 1024**2)
        hosts = [torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        for host in hosts:
            host.fill_(23)
        destinations = [
            torch.empty(nbytes, dtype=torch.uint8, device=device) for device in range(2)
        ]
        streams = [torch.cuda.Stream(device=device) for device in range(2)]
        iterations = iterations_for(nbytes, target_gib=4.0)

        def launch(
            local_streams: list[torch.cuda.Stream] = streams,
            local_destinations: list[torch.Tensor] = destinations,
            local_hosts: list[torch.Tensor] = hosts,
        ) -> None:
            for device in range(2):
                with torch.cuda.device(device), torch.cuda.stream(local_streams[device]):
                    local_destinations[device].copy_(local_hosts[device], non_blocking=True)

        for _ in range(4):
            launch()
        for device in range(2):
            torch.cuda.synchronize(device)

        elapsed_ms: list[float] = []
        for _ in range(7):
            started = time.perf_counter()
            for _ in range(iterations):
                launch()
            for device in range(2):
                torch.cuda.synchronize(device)
            elapsed_ms.append((time.perf_counter() - started) * 1000.0 / iterations)

        rates = [2 * nbytes / (value / 1000.0) / 1e9 for value in elapsed_ms]
        rows.append(
            {
                "size_mib_per_gpu": size_mib,
                "iterations_per_round": iterations,
                "makespan_ms": summary(elapsed_ms),
                "aggregate_payload_gbps": summary(rates),
                "per_gpu_payload_gbps": statistics.median(rates) / 2,
            }
        )
        del hosts, destinations, streams
    return rows


def p2p_bandwidth(sizes_mib: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "can_access_peer_0_to_1": torch.cuda.can_device_access_peer(0, 1),
        "can_access_peer_1_to_0": torch.cuda.can_device_access_peer(1, 0),
        "unidirectional": [],
        "bidirectional": [],
    }
    if not result["can_access_peer_0_to_1"]:
        return result

    for size_mib in sizes_mib:
        nbytes = round(size_mib * 1024**2)
        source0 = torch.empty(nbytes, dtype=torch.uint8, device=0)
        source1 = torch.empty(nbytes, dtype=torch.uint8, device=1)
        dest0 = torch.empty_like(source0)
        dest1 = torch.empty_like(source1)
        streams = [torch.cuda.Stream(device=device) for device in range(2)]
        iterations = iterations_for(nbytes, target_gib=8.0)

        def one_way(
            local_streams: list[torch.cuda.Stream] = streams,
            source: torch.Tensor = source0,
            destination: torch.Tensor = dest1,
        ) -> None:
            with torch.cuda.device(1), torch.cuda.stream(local_streams[1]):
                destination.copy_(source, non_blocking=True)

        def both_ways(
            local_streams: list[torch.cuda.Stream] = streams,
            local_source0: torch.Tensor = source0,
            local_source1: torch.Tensor = source1,
            local_dest0: torch.Tensor = dest0,
            local_dest1: torch.Tensor = dest1,
        ) -> None:
            with torch.cuda.device(1), torch.cuda.stream(local_streams[1]):
                local_dest1.copy_(local_source0, non_blocking=True)
            with torch.cuda.device(0), torch.cuda.stream(local_streams[0]):
                local_dest0.copy_(local_source1, non_blocking=True)

        for mode, fn, payload_factor in (
            ("unidirectional", one_way, 1),
            ("bidirectional", both_ways, 2),
        ):
            for _ in range(4):
                fn()
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            elapsed_ms: list[float] = []
            for _ in range(7):
                started = time.perf_counter()
                for _ in range(iterations):
                    fn()
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
                elapsed_ms.append((time.perf_counter() - started) * 1000.0 / iterations)
            rates = [payload_factor * nbytes / (value / 1000.0) / 1e9 for value in elapsed_ms]
            result[mode].append(
                {
                    "size_mib": size_mib,
                    "latency_ms": summary(elapsed_ms),
                    "aggregate_payload_gbps": summary(rates),
                }
            )
        del source0, source1, dest0, dest1, streams
    return result


def fp8_dequant(device: int, values: int = 128 * 1024 * 1024) -> dict[str, Any]:
    result: dict[str, Any] = {"values": values, "block_size": 128}
    if not hasattr(torch, "float8_e4m3fn"):
        result["error"] = "torch.float8_e4m3fn is unavailable"
        return result
    try:
        with torch.cuda.device(device):
            bits = torch.randint(0, 120, (values,), dtype=torch.uint8, device=device)
            source = bits.view(torch.float8_e4m3fn)
            scales = torch.ones(values // 128, dtype=torch.float32, device=device)
            output = torch.empty(values, dtype=torch.bfloat16, device=device)
            nbytes_useful = values * (1.0 + 4.0 / 128.0 + 2.0)
            iterations = iterations_for(round(nbytes_useful), target_gib=6.0)

            cast_ms = cuda_event_ms(
                lambda: output.copy_(source), device=device, iterations=iterations
            )

            def cast_and_scale() -> None:
                output.copy_(source)
                output.view(-1, 128).mul_(scales[:, None])

            scaled_ms = cuda_event_ms(cast_and_scale, device=device, iterations=iterations)

        def convert(times: list[float], physical_bytes_per_value: float) -> dict[str, Any]:
            value_rates = [values / (value / 1000.0) / 1e9 for value in times]
            physical_rates = [
                values * physical_bytes_per_value / (value / 1000.0) / 1e9 for value in times
            ]
            return {
                "latency_ms": summary(times),
                "gvalues_per_second": summary(value_rates),
                "estimated_physical_gbps": summary(physical_rates),
            }

        result["cast_only"] = convert(cast_ms, 3.0)
        # Cast: read 1 B/write 2 B. Scale: read 2 B + 4/128 B/write 2 B.
        result["cast_then_block_scale"] = convert(scaled_ms, 7.0 + 4.0 / 128.0)
    except Exception as error:  # noqa: BLE001 - benchmark records unsupported paths
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def gemm_profile(device: int) -> list[dict[str, Any]]:
    shapes = [
        ("glm-column-tp2", 6144, 3072),
        ("glm-expert-up-tp2", 6144, 2048),
        ("kimi-column-tp2", 7168, 3584),
        ("kimi-expert-up-tp2", 7168, 2048),
        ("v4-column-tp4", 4096, 1024),
    ]
    batches = (1, 8, 32)
    rows: list[dict[str, Any]] = []
    with torch.cuda.device(device):
        for name, k, n in shapes:
            one_weight_bytes = k * n * 2
            copies = max(2, min(6, math.ceil(192 * 1024**2 / one_weight_bytes)))
            weights = [
                torch.randn(k, n, dtype=torch.bfloat16, device=device) for _ in range(copies)
            ]
            for batch in batches:
                x = torch.randn(batch, k, dtype=torch.bfloat16, device=device)
                y = torch.empty(batch, n, dtype=torch.bfloat16, device=device)
                cursor = 0

                def operation(
                    local_x: torch.Tensor = x,
                    local_weights: list[torch.Tensor] = weights,
                    local_y: torch.Tensor = y,
                ) -> None:
                    nonlocal cursor
                    torch.mm(local_x, local_weights[cursor], out=local_y)
                    cursor = (cursor + 1) % len(local_weights)

                useful_bytes = one_weight_bytes + 2 * batch * (k + n)
                iterations = iterations_for(useful_bytes, target_gib=8.0)
                times = cuda_event_ms(
                    operation, device=device, iterations=iterations, warmup=copies
                )
                tflops = [2 * batch * k * n / (value / 1000.0) / 1e12 for value in times]
                gbps = [useful_bytes / (value / 1000.0) / 1e9 for value in times]
                rows.append(
                    {
                        "name": name,
                        "m": batch,
                        "k": k,
                        "n": n,
                        "rotating_weight_copies": copies,
                        "weight_mib": one_weight_bytes / 1024**2,
                        "latency_ms": summary(times),
                        "tflops": summary(tflops),
                        "mfu_fraction": statistics.median(tflops) / A100_BF16_PEAK_TFLOPS,
                        "useful_weight_bandwidth_gbps": summary(gbps),
                        "mbu_fraction": statistics.median(gbps) / A100_SXM4_80GB_PEAK_GBPS,
                    }
                )
            del weights
            torch.cuda.empty_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("run with exactly CUDA_VISIBLE_DEVICES=2,3")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_unix_seconds": time.time(),
        "host_class": "8xa100-sxm4-80gb",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_gpu_2_3": run_text(
            [
                "nvidia-smi",
                "-i",
                "2,3",
                "--query-gpu=index,name,driver_version,power.limit,clocks.max.sm,clocks.max.memory",
                "--format=csv,noheader",
            ]
        ),
        "numa": run_text(["numactl", "--show"]),
        "devices": [device_metadata(device) for device in range(2)],
        "assumed_peaks": {
            "hbm_gbps": A100_SXM4_80GB_PEAK_GBPS,
            "bf16_dense_tflops": A100_BF16_PEAK_TFLOPS,
        },
        "measurements": {},
    }

    measurements = payload["measurements"]
    assert isinstance(measurements, dict)
    measurements["hbm"] = [hbm_bandwidth(device) for device in range(2)]
    transfer_sizes = [0.25, 1, 4, 16, 64, 256]
    measurements["h2d_single"] = {
        str(device): h2d_single(device, transfer_sizes) for device in range(2)
    }
    measurements["h2d_dual"] = h2d_dual(transfer_sizes)
    measurements["p2p"] = p2p_bandwidth([0.004, 0.25, 1, 4, 16, 64, 256])
    measurements["fp8_dequant"] = [fp8_dequant(device) for device in range(2)]
    fused_dequant = run_text(["./dequant_bench"])
    try:
        measurements["fp8_dequant_cuda_fused"] = json.loads(fused_dequant)
    except json.JSONDecodeError:
        measurements["fp8_dequant_cuda_fused"] = {"error": fused_dequant}
    measurements["bf16_gemm"] = [gemm_profile(device) for device in range(2)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
