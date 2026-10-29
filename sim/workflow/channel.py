import abc
import enum
import logging
from dataclasses import dataclass
from typing import Generator, Optional, Union

import simpy
from ether.core import Node, UninterruptingFlow

logger = logging.getLogger(__name__)


class DataPassingMode(str, enum.Enum):
    """
    Data passing strategies for serverless edge workflows.
    """
    DIRECT = "direct"                  # Direct P2P transmission (Node A -> Node B)
    REMOTE_STORAGE = "remote_storage"  # Triangular transmission (Node A -> Storage -> Node B)
    COLOCATED = "colocated"            # Local memory/IPC transfer on the same node


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

    def __init__(self, mode: DataPassingMode, memory_bandwidth_bytes_per_sec: float = 1e9):
        """
        :param mode: The data passing mode.
        :param memory_bandwidth_bytes_per_sec: In-memory/IPC bandwidth for colocated transfers
                                                (default: 1 GB/s).
        """
        self.mode = mode
        self.memory_bandwidth = memory_bandwidth_bytes_per_sec
        self.transfer_records = []

    @abc.abstractmethod
    def transfer(
        self,
        env,
        source_node: Union[str, Node],
        target_node: Union[str, Node],
        size_bytes: int,
        source_task: str = "",
        target_task: str = ""
    ) -> Generator[simpy.Event, None, DataTransferRecord]:
        """
        SimPy generator that simulates intermediate data movement between source_node and target_node.
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

    def _execute_network_flow(self, env, src_node: Node, dst_node: Node, size_bytes: int) -> Generator[simpy.Event, None, None]:
        """
        Executes a network flow between src_node and dst_node using Ether.
        If src_node == dst_node, simulates local memory transfer.
        """
        if src_node == dst_node or size_bytes <= 0:
            local_duration = size_bytes / self.memory_bandwidth if size_bytes > 0 else 1e-6
            yield env.timeout(max(local_duration, 1e-6))
            return

        route = env.topology.route(src_node, dst_node)
        if env.flow_factory is not None:
            flow = env.flow_factory.create_flow(env, size_bytes, route)
        else:
            flow = UninterruptingFlow(env, size_bytes, route)

        yield flow.start()


class DirectDataChannel(DataChannel):
    """
    Direct Data Passing Channel:
    Transfers intermediate data directly from source_node to target_node across
    the network topology (peer-to-peer / point-to-point) without any intermediate storage hop.
    """

    def __init__(self, memory_bandwidth_bytes_per_sec: float = 1e9):
        super().__init__(DataPassingMode.DIRECT, memory_bandwidth_bytes_per_sec)

    def transfer(
        self,
        env,
        source_node: Union[str, Node],
        target_node: Union[str, Node],
        size_bytes: int,
        source_task: str = "",
        target_task: str = ""
    ) -> Generator[simpy.Event, None, DataTransferRecord]:
        src = self._resolve_node(env.topology, source_node)
        dst = self._resolve_node(env.topology, target_node)

        t_start = env.now
        logger.debug(
            "[DirectChannel] Transferring %d bytes from %s (%s) to %s (%s) at t=%.4f",
            size_bytes, source_task, src.name, target_task, dst.name, t_start
        )

        yield from self._execute_network_flow(env, src, dst, size_bytes)

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
            hops=1 if src != dst else 0
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

    def __init__(self, storage_node: Union[str, Node], memory_bandwidth_bytes_per_sec: float = 1e9):
        super().__init__(DataPassingMode.REMOTE_STORAGE, memory_bandwidth_bytes_per_sec)
        self.storage_node_ref = storage_node

    def transfer(
        self,
        env,
        source_node: Union[str, Node],
        target_node: Union[str, Node],
        size_bytes: int,
        source_task: str = "",
        target_task: str = ""
    ) -> Generator[simpy.Event, None, DataTransferRecord]:
        src = self._resolve_node(env.topology, source_node)
        dst = self._resolve_node(env.topology, target_node)
        storage = self._resolve_node(env.topology, self.storage_node_ref)

        t_start = env.now
        logger.debug(
            "[RemoteStorage] Hop 1: Uploading %d bytes from %s (%s) to storage %s at t=%.4f",
            size_bytes, source_task, src.name, storage.name, t_start
        )

        # Hop 1: Upload to storage
        yield from self._execute_network_flow(env, src, storage, size_bytes)

        t_mid = env.now
        logger.debug(
            "[RemoteStorage] Hop 2: Downloading %d bytes from storage %s to %s (%s) at t=%.4f",
            size_bytes, storage.name, target_task, dst.name, t_mid
        )

        # Hop 2: Download from storage to destination
        yield from self._execute_network_flow(env, storage, dst, size_bytes)

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
            hops=2
        )
        self.transfer_records.append(record)
        return record


class ColocatedDataChannel(DataChannel):
    """
    Colocated Channel (Same-Node / VM Storage):
    Assumes functions are scheduled on the same VM/host. Data transfer is
    achieved via in-memory buffer or local volume IPC.
    If source and target are placed on different nodes, raises an error.
    """

    def __init__(self, memory_bandwidth_bytes_per_sec: float = 1e9, strict: bool = True):
        super().__init__(DataPassingMode.COLOCATED, memory_bandwidth_bytes_per_sec)
        self.strict = strict

    def transfer(
        self,
        env,
        source_node: Union[str, Node],
        target_node: Union[str, Node],
        size_bytes: int,
        source_task: str = "",
        target_task: str = ""
    ) -> Generator[simpy.Event, None, DataTransferRecord]:
        src = self._resolve_node(env.topology, source_node)
        dst = self._resolve_node(env.topology, target_node)

        if src != dst:
            if self.strict:
                raise ValueError(
                    f"ColocatedDataChannel requires source and target to be on the same node! "
                    f"Got {src.name} != {dst.name}."
                )
            logger.warning(
                "ColocatedDataChannel placed across different nodes (%s != %s). Falling back to direct flow.",
                src.name, dst.name
            )

        t_start = env.now
        yield from self._execute_network_flow(env, src, dst, size_bytes)
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
            hops=0 if src == dst else 1
        )
        self.transfer_records.append(record)
        return record
