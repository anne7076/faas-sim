import abc
import collections
import logging
from typing import Callable, Deque, Dict, List, Optional, Tuple, Union

from ether.core import Node
from skippy.core.clustercontext import ClusterContext
from skippy.core.model import (
    Container,
    Node as SkippyNode,
    Pod,
    PodSpec,
    ResourceRequirements,
    SchedulingResult,
)
from skippy.core.predicates import Predicate
from skippy.core.priorities import Priority
from skippy.core.scheduler import Scheduler
from skippy.core.utils import normalize_image_name

from sim.core import Environment
from sim.workflow.model import WorkflowDAG, WorkflowTask, DataDependency, DataPassingMode

logger = logging.getLogger(__name__)


class WorkflowScheduler(abc.ABC):
    """
    Abstract interface for workflow-aware scheduling policies that map workflow tasks
    to physical nodes in the edge/cloud topology.
    """

    @abc.abstractmethod
    def schedule_task(
        self,
        task: WorkflowTask,
        workflow: WorkflowDAG,
        placements: Dict[str, str],
        env: Environment
    ) -> str:
        """
        Selects a target node for a single `task` just-in-time (on-time), given the
        already-resolved node placements of upstream tasks in `placements`.

        :param task: The WorkflowTask to schedule now.
        :param workflow: The WorkflowDAG containing the task and its dependencies.
        :param placements: Mapping of already-scheduled `task_name -> node_name`.
        :param env: The faas-sim simulation environment.
        :return: The selected physical `node_name`.
        """
        pass

    def schedule(self, workflow: WorkflowDAG, env: Environment) -> Dict[str, str]:
        """
        Computes placements for all tasks in topological order by delegating each
        step to `schedule_task`.
        """
        placements: Dict[str, str] = {}
        for task_name in workflow.get_execution_order():
            task = workflow.tasks[task_name]
            placements[task_name] = self.schedule_task(task, workflow, placements, env)
        return placements

    def _get_compute_nodes(self, env: Environment, task: Optional[WorkflowTask] = None) -> List[Node]:
        """
        Returns feasible compute nodes from the environment.
        When running inside a full faas-sim Simulation (env.cluster and env.scheduler exist),
        uses Skippy's predicates (CPU/RAM resources, node labels, storage exclusion) to filter
        out infeasible nodes for the given task's function deployment without mutating global pod counters.
        """
        if env.topology is None:
            return [Node("default_node")]

        all_ether_nodes = [
            n for n in env.topology.get_nodes()
            if not n.name.startswith("switch") and "registry" not in n.name
        ]
        if not all_ether_nodes:
            return [Node("default_node")]

        if (
            task is not None
            and getattr(env, "cluster", None) is not None
            and getattr(env, "scheduler", None) is not None
            and getattr(env, "faas", None) is not None
            and hasattr(env.faas, "deployment_service")
            and env.faas.deployment_service.exists(task.function_name)
        ):
            fd = env.faas.deployment_service.get_by_name(task.function_name)
            containers = fd.deployment_ranking.containers if fd.deployment_ranking else fd.fn_containers
            if containers:
                probe_pod = self._build_probe_pod(fd.name, containers[0])
                warm_node_names = {
                    r.node.name
                    for r in env.faas.get_replicas(task.function_name)
                    if r.node is not None
                }
                feasible_names = warm_node_names | {
                    s_node.name
                    for s_node in env.cluster.list_nodes()
                    if env.scheduler.passes_predicates(probe_pod, s_node)
                }
                feasible_ether_nodes = [n for n in all_ether_nodes if n.name in feasible_names]
                if feasible_ether_nodes:
                    return feasible_ether_nodes

        return all_ether_nodes

    @staticmethod
    def _build_probe_pod(fn_name: str, fn_container) -> Pod:
        """
        Constructs a lightweight Pod for Skippy predicate evaluation without mutating
        `sim.skippy.pod_counters`.
        """
        requests = fn_container.resource_config.get_resource_requirements()
        spec = PodSpec()
        spec.containers = [Container(fn_container.image, ResourceRequirements(requests))]
        spec.labels = fn_container.labels.copy()
        pod = Pod(f"probe-pod-{fn_name}", "faas-sim")
        pod.spec = spec
        return pod

    def _estimate_transfer_time(
        self,
        env: Environment,
        src_node: Node,
        dst_node: Node,
        size_bytes: int,
        mode: DataPassingMode,
        storage_node: Optional[Node] = None
    ) -> float:
        """
        Estimates the transfer time in seconds for `size_bytes` from `src_node` to `dst_node`.
        """
        if size_bytes <= 0:
            return 1e-6

        if src_node == dst_node or mode == DataPassingMode.COLOCATED:
            return size_bytes / 1e9

        if mode == DataPassingMode.REMOTE_STORAGE:
            effective_storage = storage_node or src_node
            hop1 = self._estimate_flow_time(env, src_node, effective_storage, size_bytes)
            hop2 = self._estimate_flow_time(env, effective_storage, dst_node, size_bytes)
            return hop1 + hop2

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
            rtt_latency = (route.rtt / 1000.0) / 2.0

            if route.hops:
                bottleneck_bw = min(getattr(link, "bandwidth", 1e8) for link in route.hops)
                if bottleneck_bw <= 0:
                    bottleneck_bw = 1e6
                transmission_time = size_bytes / bottleneck_bw
            else:
                transmission_time = 0.0

            return rtt_latency + transmission_time
        except Exception as e:
            logger.warning("[WorkflowScheduler] Failed to route between %s and %s: %s", src.name, dst.name, e)
            return 0.1

    @staticmethod
    def _resolve_node(env: Environment, node_ref: Union[str, Node]) -> Optional[Node]:
        if isinstance(node_ref, Node):
            return node_ref
        if env.topology is not None:
            return env.topology.find_node(node_ref)
        return None


class RoundRobinWorkflowScheduler(WorkflowScheduler):
    """
    Baseline workflow-agnostic scheduler:
    Places tasks in round-robin fashion across feasible compute nodes without
    considering data dependencies, payload sizes, or network topology.
    """

    def schedule_task(
        self,
        task: WorkflowTask,
        workflow: WorkflowDAG,
        placements: Dict[str, str],
        env: Environment
    ) -> str:
        available_nodes = self._get_compute_nodes(env, task)
        idx = len(placements)
        return available_nodes[idx % len(available_nodes)].name


class DataAwareWorkflowScheduler(WorkflowScheduler):
    """
    Data-Aware Workflow Scheduler:
    Optimizes function placement across edge and cloud nodes by taking into account
    intermediate data payload sizes, data passing modes, and network topology
    characteristics (bandwidth, latency/RTT).

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
        cold_start_penalty: float = 0.5,
        max_tasks_per_node: Optional[int] = None
    ):
        self.storage_node_ref: Optional[Union[str, Node]] = storage_node
        self.network_weight: float = network_weight
        self.load_balance_weight: float = load_balance_weight
        self.cold_start_penalty: float = cold_start_penalty
        self.max_tasks_per_node: Optional[int] = max_tasks_per_node

    def schedule_task(
        self,
        task: WorkflowTask,
        workflow: WorkflowDAG,
        placements: Dict[str, str],
        env: Environment
    ) -> str:
        candidate_nodes = self._get_compute_nodes(env, task)
        incoming_deps = workflow.get_incoming_dependencies(task.name)
        storage_node = self._resolve_node(env, self.storage_node_ref) if self.storage_node_ref else None

        # Identify nodes that already host a warm RUNNING replica for this task's function
        warm_nodes = set()
        if getattr(env, "faas", None) is not None and hasattr(env.faas, "get_replicas"):
            warm_nodes = {
                r.node.name
                for r in env.faas.get_replicas(task.function_name)
                if r.node is not None
            }

        # Count tasks already placed on each node in this workflow
        node_task_count: Dict[str, int] = collections.Counter(placements.values())

        eligible_candidates = [
            n for n in candidate_nodes
            if self.max_tasks_per_node is None or node_task_count[n.name] < self.max_tasks_per_node
        ]
        if not eligible_candidates:
            eligible_candidates = candidate_nodes

        # Root tasks: prefer warm nodes if available, otherwise lowest current task count
        if not incoming_deps:
            best_node = min(
                eligible_candidates,
                key=lambda n: (
                    0.0 if (not warm_nodes or n.name in warm_nodes) else self.cold_start_penalty,
                    node_task_count[n.name]
                )
            )
            logger.debug("[DataAwareScheduler] Placed root task '%s' on '%s'", task.name, best_node.name)
            return best_node.name

        best_node = None
        best_score = float("inf")

        for candidate in eligible_candidates:
            total_transfer_cost = 0.0

            for dep in incoming_deps:
                parent_node_name = placements.get(dep.source_task)
                if parent_node_name is None:
                    continue
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

            load_penalty = node_task_count[candidate.name] * self.load_balance_weight
            cold_penalty = (
                self.cold_start_penalty
                if (warm_nodes and candidate.name not in warm_nodes)
                else 0.0
            )
            candidate_score = (total_transfer_cost * self.network_weight) + load_penalty + cold_penalty

            if candidate_score < best_score:
                best_score = candidate_score
                best_node = candidate

        if best_node is None:
            best_node = eligible_candidates[0]

        logger.info(
            "[DataAwareScheduler] Placed task '%s' on '%s' (score: %.4f)",
            task.name, best_node.name, best_score
        )
        return best_node.name


class WorkflowSkippyScheduler(Scheduler):
    """
    Workflow-Aware Skippy Scheduler Extension.

    Extends Skippy's cluster-wide `Scheduler` (`env.scheduler` in faas-sim) so that:
    1. All Skippy hard-constraint predicates (CPU/RAM fit, node selectors, storage exclusion)
       and cluster state tracking (`place_pod_on_node`, `needed_images`) are preserved.
    2. When a pod is deployed/scaled for a workflow task, it places the pod on the node
       selected by the `WorkflowScheduler` policy (if feasible), falling back to standard
       Skippy priority scoring for non-workflow pods or if the target node is saturated.
    """

    def __init__(
        self,
        cluster_context: ClusterContext,
        percentage_of_nodes_to_score: int = 100,
        predicates: Optional[List[Predicate]] = None,
        priorities: Optional[List[Tuple[float, Priority]]] = None
    ):
        super().__init__(
            cluster_context=cluster_context,
            percentage_of_nodes_to_score=percentage_of_nodes_to_score,
            predicates=predicates,
            priorities=priorities
        )
        self._placement_queues: Dict[str, Deque[str]] = collections.defaultdict(collections.deque)

    @staticmethod
    def create(
        percentage_of_nodes_to_score: int = 100,
        predicates: Optional[List[Predicate]] = None,
        priorities: Optional[List[Tuple[float, Priority]]] = None
    ) -> Callable[[Environment], 'WorkflowSkippyScheduler']:
        """
        Factory helper for injecting into `sim.create_scheduler`:
            sim.create_scheduler = WorkflowSkippyScheduler.create()
        """
        def _factory(env: Environment) -> 'WorkflowSkippyScheduler':
            return WorkflowSkippyScheduler(
                env.cluster,
                percentage_of_nodes_to_score=percentage_of_nodes_to_score,
                predicates=predicates,
                priorities=priorities
            )
        return _factory

    def enqueue_placement(self, function_name: str, node_name: str) -> None:
        """
        Registers a target node decision from the WorkflowScheduler for the next pod
        of `function_name` that enters Skippy's scheduling queue.
        """
        self._placement_queues[function_name].append(node_name)

    def schedule(self, pod: Pod) -> SchedulingResult:
        """
        Schedules a Pod onto a cluster node.
        If a workflow placement target is queued for this pod's function and passes
        Skippy's predicates, places the pod on that target node; otherwise delegates
        to standard Skippy scheduling.
        """
        fn_name = self._extract_function_name(pod)
        queue = self._placement_queues.get(fn_name)

        if queue:
            target_node_name = queue.popleft()
            nodes = self.cluster_context.list_nodes()
            feasible_nodes = [node for node in nodes if self.passes_predicates(pod, node)]

            target_host: Optional[SkippyNode] = next(
                (node for node in feasible_nodes if node.name == target_node_name),
                None
            )

            if target_host is not None:
                needed_images = []
                host_images = self.cluster_context.images_on_nodes[target_host.name]
                for container in pod.spec.containers:
                    if normalize_image_name(container.image) not in host_images:
                        needed_images.append(normalize_image_name(container.image))

                self.cluster_context.place_pod_on_node(pod, target_host)
                logger.info(
                    "[WorkflowSkippyScheduler] Placed workflow pod '%s' onto target node '%s'",
                    pod.name, target_host.name
                )
                return SchedulingResult(
                    suggested_host=target_host,
                    feasible_nodes=len(feasible_nodes),
                    needed_images=needed_images
                )

            logger.warning(
                "[WorkflowSkippyScheduler] Target node '%s' for pod '%s' failed predicates or not found; "
                "falling back to standard Skippy scoring.",
                target_node_name, pod.name
            )

        return super().schedule(pod)

    @staticmethod
    def _extract_function_name(pod: Pod) -> str:
        """
        Extracts the function deployment name from a faas-sim Pod name (`pod-{fn_name}-{cnt}`).
        """
        if pod.name.startswith("pod-"):
            parts = pod.name.split("-")
            if len(parts) >= 3:
                return "-".join(parts[1:-1])
        return pod.name
