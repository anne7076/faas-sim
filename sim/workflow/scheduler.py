import abc
import logging
from typing import Dict, List, Optional, Union

from ether.core import Node

from sim.core import Environment
from sim.workflow.model import WorkflowDAG, WorkflowTask, DataDependency, DataPassingMode

logger = logging.getLogger(__name__)


class WorkflowScheduler(abc.ABC):
    """
    Abstract interface for workflow-aware schedulers that map workflow tasks
    to nodes in the physical edge/cloud topology.
    """

    @abc.abstractmethod
    def schedule(self, workflow: WorkflowDAG, env: Environment) -> Dict[str, str]:
        """
        Computes placement for each task in the workflow.

        :param workflow: The WorkflowDAG to schedule.
        :param env: The faas-sim simulation environment (provides topology, nodes).
        :return: A dictionary mapping task_name -> node_name.
        """
        pass


class RoundRobinWorkflowScheduler(WorkflowScheduler):
    """
    Baseline workflow-agnostic scheduler:
    Places tasks in round-robin fashion across available compute nodes without
    considering data dependencies, payload sizes, or network topology.
    """

    def schedule(self, workflow: WorkflowDAG, env: Environment) -> Dict[str, str]:
        available_nodes = self._get_compute_nodes(env)
        placements: Dict[str, str] = {}

        for idx, task_name in enumerate(workflow.get_execution_order()):
            node = available_nodes[idx % len(available_nodes)]
            placements[task_name] = node.name

        return placements

    def _get_compute_nodes(self, env: Environment) -> List[Node]:
        if env.topology is not None:
            nodes = [n for n in env.topology.get_nodes() if not n.name.startswith("switch")]
            if nodes:
                return nodes
        return [Node("default_node")]


class DataAwareWorkflowScheduler(WorkflowScheduler):
    """
    Data-Aware Workflow Scheduler:
    Optimizes function placement across edge and cloud nodes by taking into account
    intermediate data payload sizes, data passing modes (Direct, Remote Storage, Colocated),
    and network topology characteristics (bandwidth, latency/RTT).

    Placement Objective:
      Minimizes intermediate data transfer latency by co-locating tightly coupled tasks
      or placing them on nodes connected by high-bandwidth / low-latency direct links,
      with optional load-balancing weighting.
    """

    def __init__(
        self,
        storage_node: Optional[Union[str, Node]] = None,
        network_weight: float = 1.0,
        load_balance_weight: float = 0.0,
        max_tasks_per_node: Optional[int] = None
    ):
        """
        :param storage_node: Central storage node reference (for Remote Storage estimates).
        :param network_weight: Weight given to network transfer cost in candidate scoring.
        :param load_balance_weight: Weight given to spreading load across nodes (penalty per existing task).
        :param max_tasks_per_node: Optional strict limit on tasks allowed per node.
        """
        self.storage_node_ref: Optional[Union[str, Node]] = storage_node
        self.network_weight: float = network_weight
        self.load_balance_weight: float = load_balance_weight
        self.max_tasks_per_node: Optional[int] = max_tasks_per_node

    def schedule(self, workflow: WorkflowDAG, env: Environment) -> Dict[str, str]:
        execution_order = workflow.get_execution_order()
        candidate_nodes = self._get_compute_nodes(env)
        placements: Dict[str, str] = {}
        node_task_count: Dict[str, int] = {n.name: 0 for n in candidate_nodes}

        # Resolve storage node if specified
        storage_node = self._resolve_node(env, self.storage_node_ref) if self.storage_node_ref else None

        for task_name in execution_order:
            incoming_deps = workflow.get_incoming_dependencies(task_name)

            # Filter candidates respecting max_tasks_per_node if specified
            eligible_candidates = [
                n for n in candidate_nodes
                if self.max_tasks_per_node is None or node_task_count[n.name] < self.max_tasks_per_node
            ]
            if not eligible_candidates:
                eligible_candidates = candidate_nodes  # Fallback if all nodes reached threshold

            # Root tasks: place on the candidate with the lowest current load
            if not incoming_deps:
                best_node = min(eligible_candidates, key=lambda n: node_task_count[n.name])
                placements[task_name] = best_node.name
                node_task_count[best_node.name] += 1
                logger.debug("[DataAwareScheduler] Placed root task '%s' on '%s'", task_name, best_node.name)
                continue

            # Dependent tasks: evaluate each candidate node for network transfer cost + load penalty
            best_node = None
            best_score = float("inf")

            for candidate in eligible_candidates:
                total_transfer_cost = 0.0

                for dep in incoming_deps:
                    parent_node_name = placements[dep.source_task]
                    parent_node = (
                        env.topology.find_node(parent_node_name)
                        if env.topology is not None else None
                    )
                    if parent_node is None:
                        parent_node = Node(parent_node_name)

                    mode = dep.mode or workflow.default_mode

                    cost = self._estimate_transfer_time(
                        env=env,
                        src_node=parent_node,
                        dst_node=candidate,
                        size_bytes=dep.size_bytes,
                        mode=mode,
                        storage_node=storage_node
                    )
                    total_transfer_cost += cost

                # Composite score: network transfer cost + load balancing penalty
                load_penalty = node_task_count[candidate.name] * self.load_balance_weight
                candidate_score = (total_transfer_cost * self.network_weight) + load_penalty

                logger.debug(
                    "[DataAwareScheduler] Task '%s' candidate '%s': transfer=%.6fs, load_penalty=%.4f -> score=%.6f",
                    task_name, candidate.name, total_transfer_cost, load_penalty, candidate_score
                )

                if candidate_score < best_score:
                    best_score = candidate_score
                    best_node = candidate

            if best_node is None:
                best_node = eligible_candidates[0]

            placements[task_name] = best_node.name
            node_task_count[best_node.name] += 1
            logger.info(
                "[DataAwareScheduler] Placed task '%s' on '%s' (score: %.4f)",
                task_name, best_node.name, best_score
            )

        return placements

    def _estimate_transfer_time(
        self,
        env: Environment,
        src_node: Node,
        dst_node: Node,
        size_bytes: int,
        mode: DataPassingMode,
        storage_node: Optional[Node]
    ) -> float:
        """
        Estimates the transfer time in seconds for size_bytes from src_node to dst_node.
        """
        if size_bytes <= 0:
            return 1e-6

        # Case 1: Colocated (same node)
        if src_node == dst_node or mode == DataPassingMode.COLOCATED:
            # In-memory bandwidth approx 1 GB/s
            return size_bytes / 1e9

        # Case 2: Remote Storage (2 hops: src -> storage -> dst)
        if mode == DataPassingMode.REMOTE_STORAGE:
            if storage_node is None:
                # Default fallback penalty if no storage node defined
                storage_node = src_node
            hop1 = self._estimate_flow_time(env, src_node, storage_node, size_bytes)
            hop2 = self._estimate_flow_time(env, storage_node, dst_node, size_bytes)
            return hop1 + hop2

        # Case 3: Direct Peer-to-Peer (1 hop: src -> dst)
        return self._estimate_flow_time(env, src_node, dst_node, size_bytes)

    def _estimate_flow_time(self, env: Environment, src: Node, dst: Node, size_bytes: int) -> float:
        """
        Estimates flow duration across Ether network topology:
          Duration = (RTT / 2) + (size_bytes / bottleneck_bandwidth)
        """
        if src == dst or env.topology is None:
            return size_bytes / 1e9

        try:
            route = env.topology.route(src, dst)
            rtt_latency = (route.rtt / 1000.0) / 2.0  # One-way latency in seconds

            if route.hops:
                # In ether, link bandwidth is in bytes/second
                bottleneck_bw = min(getattr(link, 'bandwidth', 1e8) for link in route.hops)
                if bottleneck_bw <= 0:
                    bottleneck_bw = 1e6
                transmission_time = size_bytes / bottleneck_bw
            else:
                transmission_time = 0.0

            return rtt_latency + transmission_time
        except Exception as e:
            logger.warning("[DataAwareScheduler] Failed to route between %s and %s: %s", src.name, dst.name, e)
            return 0.1  # Fallback estimate

    def _get_compute_nodes(self, env: Environment) -> List[Node]:
        """Filters compute nodes from the topology (excluding switches)."""
        if env.topology is not None:
            nodes = [n for n in env.topology.get_nodes() if not n.name.startswith("switch")]
            if nodes:
                return nodes
        return [Node("default_node")]

    def _resolve_node(self, env: Environment, node_ref: Union[str, Node]) -> Optional[Node]:
        if isinstance(node_ref, Node):
            return node_ref
        if env.topology is not None:
            return env.topology.find_node(node_ref)
        return None
