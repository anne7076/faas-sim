import abc
import logging
from dataclasses import dataclass
from typing import Generator, Optional, Tuple, Union, Dict, List

import simpy
from ether.core import Node, UninterruptingFlow

from sim.core import Environment
from sim.workflow.model import DataPassingMode

logger = logging.getLogger(__name__)


@dataclass
class DataTransferRecord:
    """
    Telemetry record of an intermediate data transfer event between workflow tasks.
    """
    source_task: str
    target_task: str
    source_node: str
    target_node: str
    size_bytes: int
    start_time: float
    end_time: float
    duration: float
    mode: DataPassingMode
    hops: int = 1


class DataChannel(abc.ABC):
    """
    Abstract interface for data transfer channels between workflow function replicas.
    """

    def __init__(
        self,
        mode: DataPassingMode,
        memory_bandwidth_bytes_per_sec: float = 1e9,
        min_memory_latency: float = 1e-6
    ):
        """
        :param mode: The data passing mode.
        :param memory_bandwidth_bytes_per_sec: In-memory/IPC bandwidth for colocated transfers
                                                (default: 1 GB/s).
        :param min_memory_latency: Minimum non-zero latency (in seconds) for in-memory / local IPC
                                   transfers (default: 1 microsecond = 1e-6 s). This reflects real
                                   syscall/buffer allocation overhead and prevents zero-duration
                                   event ambiguity in SimPy.
        """
        self.mode: DataPassingMode = mode
        self.memory_bandwidth: float = memory_bandwidth_bytes_per_sec
        self.min_memory_latency: float = min_memory_latency
        self.transfer_records: List[DataTransferRecord] = []

    @abc.abstractmethod
    def transfer(
        self,
        env: Environment,
        source_node: Union[str, Node],
        target_node: Union[str, Node],
        size_bytes: int,
        source_task: str = "",
        target_task: str = "",
        shared_uploads: Optional[Dict[Tuple[str, str, str, int], simpy.Event]] = None
    ) -> Generator[simpy.Event, None, DataTransferRecord]:
        """
        SimPy generator simulating intermediate data movement between source_node and target_node.
        """
        pass

    def _resolve_node(self, topology, node: Union[str, Node]) -> Node:
        """Resolves node name or Node instance to an ether.core.Node in the topology."""
        if isinstance(node, Node):
            return node
        found = topology.find_node(node)
        if found is None:
            raise ValueError(f"Node '{node}' was not found in the topology.")
        return found

    def _execute_network_flow(
        self,
        env: Environment,
        src_node: Node,
        dst_node: Node,
        size_bytes: int,
        flow_tag: str = "workflow_data"
    ) -> Generator[simpy.Event, None, None]:
        """
        Executes a network flow between src_node and dst_node using Ether.
        If src_node == dst_node, simulates local in-memory IPC transfer.
        Also records telemetry in env.metrics if available.
        """
        if src_node == dst_node or size_bytes <= 0:
            local_duration = size_bytes / self.memory_bandwidth if size_bytes > 0 else self.min_memory_latency
            yield env.timeout(max(local_duration, self.min_memory_latency))
            return

        t_start = env.now
        route = env.topology.route(src_node, dst_node)
        if env.flow_factory is not None:
            flow = env.flow_factory.create_flow(env, size_bytes, route)
        else:
            flow = UninterruptingFlow(env, size_bytes, route)

        yield flow.start()

        # Telemetry: integrate with faas-sim global metrics if enabled
        if getattr(env, "metrics", None) is not None:
            flow_duration = env.now - t_start
            for hop in route.hops:
                env.metrics.log_network(size_bytes, flow_tag, hop)
            env.metrics.log_flow(size_bytes, flow_duration, route.source, route.destination, flow_tag)


class DirectDataChannel(DataChannel):
    """
    Direct Data Passing Channel:
    Transfers intermediate data directly from source_node to target_node across
    the network topology (peer-to-peer / point-to-point) without any intermediate storage hop.
    """

    def __init__(
        self,
        memory_bandwidth_bytes_per_sec: float = 1e9,
        min_memory_latency: float = 1e-6
    ):
        super().__init__(DataPassingMode.DIRECT, memory_bandwidth_bytes_per_sec, min_memory_latency)

    def transfer(
        self,
        env: Environment,
        source_node: Union[str, Node],
        target_node: Union[str, Node],
        size_bytes: int,
        source_task: str = "",
        target_task: str = "",
        shared_uploads: Optional[Dict[Tuple[str, str, str, int], simpy.Event]] = None
    ) -> Generator[simpy.Event, None, DataTransferRecord]:
        src: Node = self._resolve_node(env.topology, source_node)
        dst: Node = self._resolve_node(env.topology, target_node)

        t_start = env.now
        logger.debug(
            "[DirectChannel] Transferring %d bytes from %s (%s) to %s (%s) at t=%.4f",
            size_bytes, source_task, src.name, target_task, dst.name, t_start
        )

        yield from self._execute_network_flow(env, src, dst, size_bytes, flow_tag="direct_passing")

        t_end = env.now
        effective_mode = DataPassingMode.COLOCATED if src == dst else self.mode
        record = DataTransferRecord(
            source_task=source_task,
            target_task=target_task,
            source_node=src.name,
            target_node=dst.name,
            size_bytes=size_bytes,
            start_time=t_start,
            end_time=t_end,
            duration=t_end - t_start,
            mode=effective_mode,
            hops=0 if src == dst else 1
        )
        self.transfer_records.append(record)
        return record


class RemoteStorageDataChannel(DataChannel):
    """
    Remote Storage Channel (Triangular Baseline):
    Simulates standard cloud-like state persistence:
      Hop 1: Upload from source_node to storage_node (e.g. S3 / MinIO / Redis).
      Hop 2: Download from storage_node to target_node.
    """

    def __init__(
        self,
        storage_node: Union[str, Node],
        memory_bandwidth_bytes_per_sec: float = 1e9,
        min_memory_latency: float = 1e-6
    ):
        super().__init__(DataPassingMode.REMOTE_STORAGE, memory_bandwidth_bytes_per_sec, min_memory_latency)
        self.storage_node_ref: Union[str, Node] = storage_node

    def transfer(
        self,
        env: Environment,
        source_node: Union[str, Node],
        target_node: Union[str, Node],
        size_bytes: int,
        source_task: str = "",
        target_task: str = "",
        shared_uploads: Optional[Dict[Tuple[str, str, str, int], simpy.Event]] = None
    ) -> Generator[simpy.Event, None, DataTransferRecord]:
        src: Node = self._resolve_node(env.topology, source_node)
        dst: Node = self._resolve_node(env.topology, target_node)
        storage: Node = self._resolve_node(env.topology, self.storage_node_ref)

        t_start = env.now
        logger.debug(
            "[RemoteStorage] Hop 1 (Upload): %d bytes from %s (%s) to storage %s at t=%.4f",
            size_bytes, source_task, src.name, storage.name, t_start
        )

        # Hop 1: Upload from worker to storage (deduplicated if multiple consumers share `shared_uploads`)
        upload_hops = 0 if src == storage else 1
        if shared_uploads is not None and source_task:
            upload_key = (source_task, src.name, storage.name, size_bytes)
            if upload_key not in shared_uploads:
                upload_done = env.event()
                shared_uploads[upload_key] = upload_done
                yield from self._execute_network_flow(env, src, storage, size_bytes, flow_tag="storage_upload")
                upload_done.succeed()
            else:
                upload_hops = 0
                yield shared_uploads[upload_key]
        else:
            yield from self._execute_network_flow(env, src, storage, size_bytes, flow_tag="storage_upload")

        t_mid = env.now
        logger.debug(
            "[RemoteStorage] Hop 2 (Download): %d bytes from storage %s to %s (%s) at t=%.4f",
            size_bytes, storage.name, target_task, dst.name, t_mid
        )

        # Hop 2: Download from storage to worker
        download_hops = 0 if storage == dst else 1
        yield from self._execute_network_flow(env, storage, dst, size_bytes, flow_tag="storage_download")

        t_end = env.now
        record = DataTransferRecord(
            source_task=source_task,
            target_task=target_task,
            source_node=src.name,
            target_node=dst.name,
            size_bytes=size_bytes,
            start_time=t_start,
            end_time=t_end,
            duration=t_end - t_start,
            mode=self.mode,
            hops=upload_hops + download_hops
        )
        self.transfer_records.append(record)
        return record


class ColocatedDataChannel(DirectDataChannel):
    """
    Colocated Channel (Same-Node / VM Storage):
    Enforces that source and target are on the same host (when `strict=True`),
    then delegates to DirectDataChannel's in-memory transfer path.
    """

    def __init__(
        self,
        memory_bandwidth_bytes_per_sec: float = 1e9,
        min_memory_latency: float = 1e-6,
        strict: bool = True
    ):
        super().__init__(memory_bandwidth_bytes_per_sec, min_memory_latency)
        self.mode = DataPassingMode.COLOCATED
        self.strict: bool = strict

    def transfer(
        self,
        env: Environment,
        source_node: Union[str, Node],
        target_node: Union[str, Node],
        size_bytes: int,
        source_task: str = "",
        target_task: str = "",
        shared_uploads: Optional[Dict[Tuple[str, str, str, int], simpy.Event]] = None
    ) -> Generator[simpy.Event, None, DataTransferRecord]:
        src: Node = self._resolve_node(env.topology, source_node)
        dst: Node = self._resolve_node(env.topology, target_node)

        if src != dst and self.strict:
            raise ValueError(
                f"ColocatedDataChannel requires source and target to be on the same node! "
                f"Got {src.name} != {dst.name}."
            )
        return (yield from super().transfer(env, src, dst, size_bytes, source_task, target_task))


class DataTransferController:
    """
    Data Transfer Controller:
    Coordinates and routes data transfers across different channels. Supports
    heterogeneous/hybrid workflows where individual dependency edges use different
    data passing strategies (Direct, Remote Storage, Colocated).
    """

    def __init__(
        self,
        storage_node: Optional[Union[str, Node]] = None,
        memory_bandwidth_bytes_per_sec: float = 1e9,
        min_memory_latency: float = 1e-6
    ):
        self.channels: Dict[DataPassingMode, DataChannel] = {
            DataPassingMode.DIRECT: DirectDataChannel(
                memory_bandwidth_bytes_per_sec=memory_bandwidth_bytes_per_sec,
                min_memory_latency=min_memory_latency
            ),
            DataPassingMode.COLOCATED: ColocatedDataChannel(
                memory_bandwidth_bytes_per_sec=memory_bandwidth_bytes_per_sec,
                min_memory_latency=min_memory_latency,
                strict=False
            ),
        }
        if storage_node is not None:
            self.channels[DataPassingMode.REMOTE_STORAGE] = RemoteStorageDataChannel(
                storage_node=storage_node,
                memory_bandwidth_bytes_per_sec=memory_bandwidth_bytes_per_sec,
                min_memory_latency=min_memory_latency
            )

    def register_channel(self, mode: DataPassingMode, channel: DataChannel):
        """Registers or overrides a channel for a specific passing mode."""
        self.channels[mode] = channel

    def transfer(
        self,
        env: Environment,
        mode: DataPassingMode,
        source_node: Union[str, Node],
        target_node: Union[str, Node],
        size_bytes: int,
        source_task: str = "",
        target_task: str = "",
        shared_uploads: Optional[Dict[Tuple[str, str, str, int], simpy.Event]] = None
    ) -> Generator[simpy.Event, None, DataTransferRecord]:
        """
        Routes the data transfer to the designated DataChannel.
        """
        channel = self.channels.get(mode)
        if channel is None:
            raise ValueError(
                f"No channel registered for data passing mode '{mode}'. "
                f"Available modes: {list(self.channels.keys())}"
            )
        kwargs = {}
        if isinstance(channel, RemoteStorageDataChannel) and shared_uploads is not None:
            kwargs["shared_uploads"] = shared_uploads
        return (yield from channel.transfer(
            env=env,
            source_node=source_node,
            target_node=target_node,
            size_bytes=size_bytes,
            source_task=source_task,
            target_task=target_task,
            **kwargs
        ))

    def get_all_records(self) -> List[DataTransferRecord]:
        """Aggregates transfer records across all managed channels."""
        records = []
        for channel in self.channels.values():
            records.extend(channel.transfer_records)
        return sorted(records, key=lambda r: r.start_time)
