# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar

from prometheus_client import Counter, Gauge, Histogram

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.logger import init_logger

PromMetric: TypeAlias = Gauge | Counter | Histogram
PromMetricT = TypeVar("PromMetricT", bound=PromMetric)

logger = init_logger(__name__)


@dataclass
class KVConnectorStats:
    """Base class for KV Connector Stats container.

    Container for transfer performance metrics or telemetry from the connector.
    All subclasses must be serializable as stats are transmitted across worker,
    scheduler, and logger processes.

    Lifecycle:
        1. Telemetry recorded by workers via `record_*()`.
        2. Aggregated across worker ranks/steps using `aggregate()`.
        3. Formatted for CLI logging via `reduce()` or recorded to Prometheus
           via `KVConnectorPromMetrics.observe()`.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def reset(self):
        """Reset the stats, clear the state."""
        raise NotImplementedError

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        """Aggregate stats with another `KVConnectorStats` object."""
        raise NotImplementedError

    def reduce(self) -> dict[str, int | float]:
        """Reduce collected observations to representative summary values.

        Meant to be called by the logger to produce a summary dictionary for
        the last logging interval.
        """
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Return True if the stats container is empty."""
        raise NotImplementedError


class KVConnectorLogging:
    """Handles accumulation and periodic CLI logging of KV connector stats.

    Receives transfer stats via `observe()`, accumulates them over the logging
    interval using `aggregate()`, and logs reduced metrics via `log()`.
    """

    def __init__(self, kv_transfer_config: KVTransferConfig | None):
        # Instantiate the connector's stats class.
        if kv_transfer_config and kv_transfer_config.kv_connector:
            self.connector_cls = KVConnectorFactory.get_connector_class(
                kv_transfer_config
            )
        self.reset()

    def reset(self):
        """Reset the internal stats accumulator."""
        self.transfer_stats_accumulator: KVConnectorStats | None = None

    def observe(self, transfer_stats_data: dict[str, Any]):
        """Accumulate new transfer stats data for the current logging interval.

        Args:
            transfer_stats_data: Dictionary payload containing stats data from
                workers or scheduler.
        """
        # Should not be called when a KVConnector is not configured.
        assert self.connector_cls is not None
        # Called periodically when connector syncs with the scheduler.
        # Note that this is not the same as the logging interval.
        # We expect transfer_stats_data to be aggregated across all workers and
        # consist of observations from a single connector or a MultiConnector.
        transfer_stats = self.connector_cls.build_kv_connector_stats(
            transfer_stats_data
        )
        if transfer_stats is None:
            logger.warning_once(
                "The connector %s is collecting stats but "
                "does not implement the "
                "`build_kv_connector_stats` method. "
                "Stats will not be logged.",
                self.connector_cls,
            )
            return

        if self.transfer_stats_accumulator is None:
            self.transfer_stats_accumulator = transfer_stats
        else:
            # Accumulate last interval stats.
            self.transfer_stats_accumulator = self.transfer_stats_accumulator.aggregate(
                transfer_stats
            )

    def log(self, log_fn=logger.info):
        """Log transfer metrics periodically, similar to throughput logging"""
        if (
            self.transfer_stats_accumulator
            and not self.transfer_stats_accumulator.is_empty()
        ):
            # Produce a single cumulative stats object for the last time
            # interval from the recorded observations.
            xfer_metrics = self.transfer_stats_accumulator.reduce()
            xfer_metrics_str = ", ".join(f"{k}={v}" for k, v in xfer_metrics.items())
            log_fn("KV Transfer metrics: %s", xfer_metrics_str)

            # Reset metrics for next interval
            self.reset()


class KVConnectorPromMetrics:
    """Base class for per-connector Prometheus metric registration and recording."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._gauge_cls = metric_types[Gauge]
        self._counter_cls = metric_types[Counter]
        self._histogram_cls = metric_types[Histogram]
        self._labelnames = labelnames
        self.per_engine_labelvalues = per_engine_labelvalues

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Record the supplied transfer statistics to Prometheus metrics.

        These statistics are engine-specific, and should be recorded to a metric
        with the appropriate 'engine' label.
        """
        raise NotImplementedError


class KVConnectorProm:
    """Support for registering and recording per-connector Prometheus metrics."""

    _gauge_cls = Gauge
    _counter_cls = Counter
    _histogram_cls = Histogram

    def __init__(
        self,
        vllm_config: VllmConfig,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.prom_metrics: KVConnectorPromMetrics | None = None
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config and kv_transfer_config.kv_connector:
            connector_cls = KVConnectorFactory.get_connector_class(kv_transfer_config)
            metric_types = {
                Gauge: self._gauge_cls,
                Counter: self._counter_cls,
                Histogram: self._histogram_cls,
            }
            self.prom_metrics = connector_cls.build_prom_metrics(
                vllm_config,
                metric_types,
                labelnames,
                per_engine_labelvalues,
            )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Delegate transfer stats observation to the registered PromMetrics."""
        if self.prom_metrics is None:
            return
        self.prom_metrics.observe(transfer_stats_data, engine_idx)
