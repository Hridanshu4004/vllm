# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics for the NIXL connector."""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

if TYPE_CHECKING:
    from vllm.distributed.nixl_utils import nixlXferTelemetry


@dataclass
class NixlKVConnectorStats(KVConnectorStats):
    """Container for NIXL KV transfer performance metrics.

     Lifecycle:
         1. Workers record transfer telemetry via `record_transfer()` into `self.data`.
         2. Worker/scheduler stats containers are merged via `aggregate()`.
         3. Periodic logging reduces stats via `reduce()` for CLI output, while
            Prometheus metrics observe raw values via `NixlPromMetrics.observe()`.

    Distributed & TP Semantics:
         Telemetry is recorded per worker rank for each NIXL transfer operation.
         - `num_successful_transfers`: Count of individual NIXL transfer handles
           completed across participating worker ranks.
         - `bytes_transferred`: Payload bytes transferred per recorded handle on a rank.
         - `Throughput (MB/s)`: Total transferred MB divided by the sum of transfer
           durations across recorded operations.

         With tensor parallelism (TP > 1), each participating TP rank can record
         its own transfer sample for a single logical request. `bytes_transferred`
         and `Avg MB per transfer` therefore reflect per-rank, per-handle payloads,
         not the full request's aggregate KV payload across all ranks. Transfer
         durations from different TP ranks may also overlap in wall-clock time,
         so summed durations should not be read as sequential elapsed time.
    """

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        # Must be serializable
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],
            "post_duration": [],
            "bytes_transferred": [],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_notifications": [],
            "num_kv_expired_reqs": [],
        }

    def record_transfer(self, res: "nixlXferTelemetry"):
        """Record telemetry for a completed NIXL transfer operation.

        Args:
            res: Telemetry object from NIXL handle:
                - `xferDuration`: Transfer duration in microseconds (us),
                  converted to seconds.
                - `postDuration`: Post-processing duration in microseconds (us),
                  converted to seconds.
                - `totalBytes`: Bytes transferred for this handle on worker rank.
                - `descCount`: Number of memory descriptors transferred.
        """
        # Keep metrics units consistent with rest of the code: time us->s
        self.data["transfer_duration"].append(res.xferDuration / 1e6)
        self.data["post_duration"].append(res.postDuration / 1e6)
        self.data["bytes_transferred"].append(res.totalBytes)
        self.data["num_descriptors"].append(res.descCount)

    def record_failed_transfer(self):
        """Record a failed NIXL transfer operation."""
        self.data["num_failed_transfers"].append(1)

    def record_failed_notification(self):
        """Record a failed NIXL notification (send_notif)."""
        self.data["num_failed_notifications"].append(1)

    def record_kv_expired_req(self):
        """Record a request that had its KV blocks expire."""
        self.data["num_kv_expired_reqs"].append(1)

    def clone_and_reset(self) -> "NixlKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        # Do not discard metrics update that are entirely failures related.
        return (
            self.num_successful_transfers == 0
            and len(self.data["num_failed_transfers"]) == 0
            and len(self.data["num_failed_notifications"]) == 0
            and len(self.data["num_kv_expired_reqs"]) == 0
        )

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        """Aggregate telemetry observations from another stats container.

        Extends internal list accumulators with observation series collected
        across worker ranks or engine steps.
        """
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        """Reduce collected telemetry observations for CLI logging output.

        Returns:
            dict[str, int | float]: Formatted summary metrics:
                - `Num successful transfers`: Total count of recorded transfer
                  operations completed across participating worker ranks.
                - `Avg xfer time (ms)`: Mean transfer duration across recorded
                  operations (ms).
                - `P90 xfer time (ms)`: 90th percentile transfer duration (ms).
                - `Avg post time (ms)`: Mean post-processing duration (ms).
                - `P90 post time (ms)`: 90th percentile post-processing
                  duration (ms).
                - `Avg MB per transfer`: Mean payload size per recorded transfer
                  operation (MB).
                - `Throughput (MB/s)`: Ratio of total MB transferred to the
                  sum of transfer durations (total_mb / total_time_seconds).
                  Summing durations reflects total bytes divided by cumulative
                  transfer time across operations rather than wall-clock
                  interval duration.
                - `Avg number of descriptors`: Mean number of descriptors per
                  transfer.
        """
        # Compute compact representative stats suitable for CLI logging
        if self.num_successful_transfers == 0:
            # CLI logging only reports successful transfers stats. If all requests in
            # the interval were unsuccessful, Prom will report failures stats instead.
            return {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg post time (ms)": 0,
                "P90 post time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
            }

        xfer_time = np.asarray(self.data["transfer_duration"])
        post_time = np.asarray(self.data["post_duration"])
        # Convert to MB for CLI logging.
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20
        descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = mb.sum()
        avg_mb = total_mb / n

        # Sum of individual transfer durations across recorded transfers
        total_time_seconds = xfer_time.sum()
        throughput_mb_s = total_mb / total_time_seconds

        return {
            "Num successful transfers": n,
            "Avg xfer time (ms)": round(xfer_time.mean() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90).item() * 1e3, 3),
            "Avg post time (ms)": round(post_time.mean() * 1e3, 3),
            "P90 post time (ms)": round(np.percentile(post_time, 90).item() * 1e3, 3),
            "Avg MB per transfer": round(avg_mb, 3),
            "Throughput (MB/s)": round(throughput_mb_s, 3),
            "Avg number of descriptors": round(descs.mean(), 1),
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


class NixlPromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics collector for NIXL KV Cache transfers.

    Registers per-engine histograms (`vllm:nixl_xfer_time_seconds`,
    `vllm:nixl_post_time_seconds`, `vllm:nixl_bytes_transferred`,
    `vllm:nixl_num_descriptors`) and counters (`vllm:nixl_num_failed_transfers`,
    `vllm:nixl_num_failed_notifications`, `vllm:nixl_num_kv_expired_reqs`).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        buckets = [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
            5.0,
        ]
        nixl_histogram_xfer_time = self._histogram_cls(
            name="vllm:nixl_xfer_time_seconds",
            documentation="Histogram of transfer duration for NIXL KV Cache transfers.",
            buckets=buckets[1:],
            labelnames=labelnames,
        )
        self.nixl_histogram_xfer_time = create_metric_per_engine(
            nixl_histogram_xfer_time, self.per_engine_labelvalues
        )
        nixl_histogram_post_time = self._histogram_cls(
            name="vllm:nixl_post_time_seconds",
            documentation="Histogram of transfer post time for NIXL KV"
            " Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_post_time = create_metric_per_engine(
            nixl_histogram_post_time, self.per_engine_labelvalues
        )
        # uniform 2kb to 16gb range
        buckets = [2 ** (10 + i) for i in range(1, 25, 2)]
        nixl_histogram_bytes_transferred = self._histogram_cls(
            name="vllm:nixl_bytes_transferred",
            documentation="Histogram of bytes transferred per NIXL KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_bytes_transferred = create_metric_per_engine(
            nixl_histogram_bytes_transferred, self.per_engine_labelvalues
        )
        buckets = [
            10,
            20,
            30,
            50,
            75,
            100,
            200,
            400,
            1000,
            2000,
            4000,
            10000,
            20000,
            50000,
        ]
        nixl_histogram_num_descriptors = self._histogram_cls(
            name="vllm:nixl_num_descriptors",
            documentation="Histogram of number of descriptors per NIXL"
            "  KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_num_descriptors = create_metric_per_engine(
            nixl_histogram_num_descriptors, self.per_engine_labelvalues
        )
        counter_nixl_num_failed_transfers = self._counter_cls(
            name="vllm:nixl_num_failed_transfers",
            documentation="Number of failed NIXL KV Cache transfers.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_failed_transfers = create_metric_per_engine(
            counter_nixl_num_failed_transfers, self.per_engine_labelvalues
        )
        counter_nixl_num_failed_notifications = self._counter_cls(
            name="vllm:nixl_num_failed_notifications",
            documentation="Number of failed NIXL KV Cache notifications.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_failed_notifications = create_metric_per_engine(
            counter_nixl_num_failed_notifications, self.per_engine_labelvalues
        )

        counter_nixl_num_kv_expired_reqs = self._counter_cls(
            name="vllm:nixl_num_kv_expired_reqs",
            documentation="Number of requests that had their KV expire. "
            "NOTE: This metric is tracked on the P instance.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_kv_expired_reqs = create_metric_per_engine(
            counter_nixl_num_kv_expired_reqs, self.per_engine_labelvalues
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Observe recorded NIXL transfer telemetry series in Prometheus metrics."""
        for prom_obj, list_item_key in zip(
            [
                self.nixl_histogram_xfer_time,
                self.nixl_histogram_post_time,
                self.nixl_histogram_bytes_transferred,
                self.nixl_histogram_num_descriptors,
            ],
            [
                "transfer_duration",
                "post_duration",
                "bytes_transferred",
                "num_descriptors",
            ],
        ):
            for list_item in transfer_stats_data[list_item_key]:
                prom_obj[engine_idx].observe(list_item)
        for counter_obj, counter_item_key in zip(
            [
                self.counter_nixl_num_failed_transfers,
                self.counter_nixl_num_failed_notifications,
                self.counter_nixl_num_kv_expired_reqs,
            ],
            ["num_failed_transfers", "num_failed_notifications", "num_kv_expired_reqs"],
        ):
            for list_item in transfer_stats_data[counter_item_key]:
                counter_obj[engine_idx].inc(list_item)
