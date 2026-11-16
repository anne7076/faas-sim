import collections
import logging
from dataclasses import dataclass, field
from typing import Callable, Counter, Dict, Generator, List, Optional, Set, Tuple, Union

import simpy
from ether.core import Node
from faas.system.core import FunctionReplicaState, FunctionRequest

from sim.core import Environment
from sim.faas import SimFunctionReplica
from sim.workflow.channel import DataTransferController, DataTransferRecord
from sim.workflow.model import DataDependency, DataPassingMode, WorkflowDAG, WorkflowTask
from sim.workflow.scheduler import DataAwareWorkflowScheduler, WorkflowScheduler, WorkflowSkippyScheduler

logger = logging.getLogger(__name__)


@dataclass
class WorkflowTaskExecutionRecord:
    """
    Telemetry record of a single task execution in a workflow.
    - start_time / end_time / duration: Pure function computation phase.
    - cold_start_duration: Time spent provisioning/polling a cold replica prior to execution.
    """
    task_name: str
    function_name: str
    node_name: str
    start_time: float
    end_time: float
    duration: float
    cold_start_duration: float = 0.0
    status: str = "SUCCESS"


@dataclass
class WorkflowExecutionResult:
    """
    Summary metrics of an executed workflow.

    - makespan: Total wall-clock time elapsed from workflow start to finish.
                Includes cold start, data transfer, compute time, and concurrency overlap.
    - total_compute_time: Arithmetic sum of pure function execution durations
                          across all tasks (excludes cold-start provisioning and network transfer time).
    - total_data_transfer_time: Arithmetic sum of all intermediate data transfer durations.
    - total_cold_start_time: Arithmetic sum of replica cold-start/provisioning durations across all tasks.
    """
    workflow_name: str
    start_time: float
    end_time: float
    makespan: float
    total_compute_time: float
    total_data_transfer_time: float
    total_data_bytes_transferred: int
    task_records: Dict[str, WorkflowTaskExecutionRecord] = field(default_factory=dict)
    transfer_records: List[DataTransferRecord] = field(default_factory=list)
    total_cold_start_time: float = 0.0

    @property
    def data_transfer_overhead_ratio(self) -> float:
        """
        Fraction of workflow makespan attributed to data transfer overhead.
        """
        if self.makespan <= 0:
            return 0.0
        return self.total_data_transfer_time / self.makespan


class WorkflowEngine:
    """
    Workflow Execution Engine (Orchestrator).

    Coordinates the execution of serverless workflow DAGs by scheduling each task
    on-time once its upstream parent tasks complete, transferring intermediate data
    from the parents' actual execution nodes, and invoking functions via faas-sim's
    native `FaasSystem` pipeline.
    """

    def __init__(
        self,
        env: Environment,
        controller: Optional[DataTransferController] = None,
        scheduler: Optional[WorkflowScheduler] = None,
        storage_node: Optional[Union[str, Node]] = None,
        default_task_duration: float = 0.5
    ):
        """
        :param env: The faas-sim Environment.
        :param controller: The DataTransferController routing data flows.
        :param scheduler: The WorkflowScheduler policy for placing tasks across topology nodes.
                          Defaults to DataAwareWorkflowScheduler(storage_node=storage_node).
        :param storage_node: Central storage node reference for remote storage transfers.
        :param default_task_duration: Fallback duration if running standalone without FaasSystem.
        """
        self.env: Environment = env
        self.controller: DataTransferController = controller or DataTransferController(storage_node=storage_node)
        self.scheduler: WorkflowScheduler = scheduler or DataAwareWorkflowScheduler(storage_node=storage_node)
        self.default_task_duration: float = default_task_duration
        self._pending_waiters: Counter[Tuple[str, str]] = collections.Counter()
        self._pending_replicas: Dict[Tuple[str, str], Optional[SimFunctionReplica]] = {}
        self._claimed_replica_ids: Set[str] = set()
        if not hasattr(self.env, "_workflow_active_node_tasks"):
            self.env._workflow_active_node_tasks = collections.Counter()

    def execute_workflow(
        self,
        workflow: WorkflowDAG,
        task_placements: Optional[Dict[str, str]] = None,
        task_executors: Optional[Dict[str, Callable[[WorkflowTask, str], Generator[simpy.Event, None, float]]]] = None
    ) -> Generator[simpy.Event, None, WorkflowExecutionResult]:
        """
        Main entry point to execute a workflow DAG in the simulation.

        1. Validates any explicit caller placement overrides.
        2. Creates completion events for each task in the DAG.
        3. Spawns tasks as concurrent SimPy processes; each task waits for its parent tasks
           to finish computing, gets scheduled on-time, resolves/deploys its replica on the
           target node (synchronizing any fallback node), pulls incoming data from the parents'
           actual execution nodes, and executes.
        4. Awaits completion of all tasks and returns execution metrics for this run.
        """
        t_start = self.env.now
        logger.info("[WorkflowEngine] Starting workflow '%s' at t=%.4f", workflow.name, t_start)

        placements = self._initialize_placements(task_placements)
        task_done_events: Dict[str, simpy.Event] = {name: self.env.event() for name in workflow.tasks}
        task_records: Dict[str, WorkflowTaskExecutionRecord] = {}
        run_transfer_records: List[DataTransferRecord] = []
        shared_uploads: Dict[Tuple[str, str, str, int], simpy.Event] = {}

        task_processes = [
            self.env.process(
                self._run_task(
                    task=task,
                    workflow=workflow,
                    placements=placements,
                    task_done_events=task_done_events,
                    task_records=task_records,
                    run_transfer_records=run_transfer_records,
                    shared_uploads=shared_uploads,
                    task_executors=task_executors
                )
            )
            for task in workflow.tasks.values()
        ]

        yield simpy.AllOf(self.env, task_processes)

        return self._compile_results(workflow, t_start, task_records, run_transfer_records)

    def _run_task(
        self,
        task: WorkflowTask,
        workflow: WorkflowDAG,
        placements: Dict[str, str],
        task_done_events: Dict[str, simpy.Event],
        task_records: Dict[str, WorkflowTaskExecutionRecord],
        run_transfer_records: List[DataTransferRecord],
        shared_uploads: Dict[Tuple[str, str, str, int], simpy.Event],
        task_executors: Optional[Dict[str, Callable]]
    ) -> Generator[simpy.Event, None, None]:
        """
        Lifecycle of a single task in the workflow:
        Step 1: Wait for all upstream parent tasks to finish computing.
        Step 2: Schedule this task on-time (all parents have finished on known nodes).
        Step 3: Resolve or deploy the target replica on `planned_node` and record `actual_node`
                (ensuring any Skippy fallback is known before data transfer begins).
        Step 4: Transfer incoming data from all parents' actual nodes to `actual_node` concurrently.
        Step 5: Invoke computation on the resolved replica, record metrics, and signal completion.
        """
        # Step 1: Wait for upstream parent computations to complete
        yield from self._wait_for_parents(task.name, workflow, task_done_events)

        # Step 2: Schedule task on-time (unless already pinned by caller override)
        planned_node = self._schedule_task_on_time(task, workflow, placements)
        active_counter: Counter[str] = getattr(self.env, "_workflow_active_node_tasks", collections.Counter())
        active_counter[planned_node] += 1
        tracked_node = planned_node

        try:
            # Step 3: Resolve or deploy replica on `planned_node` before transferring data
            target_replica, actual_node, cold_start_duration = yield from self._prepare_execution_target(
                task, planned_node, task_executors
            )
            if actual_node != tracked_node:
                active_counter[tracked_node] = max(0, active_counter[tracked_node] - 1)
                active_counter[actual_node] += 1
                tracked_node = actual_node
            placements[task.name] = actual_node

            # Step 4: Transfer incoming data from parents' actual nodes to `actual_node`
            yield from self._transfer_incoming_data(
                task=task,
                target_node=actual_node,
                workflow=workflow,
                placements=placements,
                run_transfer_records=run_transfer_records,
                shared_uploads=shared_uploads
            )

            # Step 5: Invoke pure function computation and notify downstream children
            t_task_start = self.env.now
            duration = yield from self._execute_computation(task, actual_node, target_replica, task_executors)
            t_task_end = self.env.now

            task_records[task.name] = WorkflowTaskExecutionRecord(
                task_name=task.name,
                function_name=task.function_name,
                node_name=actual_node,
                start_time=t_task_start,
                end_time=t_task_end,
                duration=duration,
                cold_start_duration=cold_start_duration
            )

            task_done_events[task.name].succeed()
        finally:
            active_counter[tracked_node] = max(0, active_counter[tracked_node] - 1)
            if active_counter[tracked_node] <= 0:
                active_counter.pop(tracked_node, None)

    def _wait_for_parents(
        self,
        task_name: str,
        workflow: WorkflowDAG,
        task_done_events: Dict[str, simpy.Event]
    ) -> Generator[simpy.Event, None, None]:
        """
        Waits for all upstream parent tasks of `task_name` to finish executing.
        """
        incoming_deps = workflow.get_incoming_dependencies(task_name)
        if incoming_deps:
            parent_events = [task_done_events[dep.source_task] for dep in incoming_deps]
            yield simpy.AllOf(self.env, parent_events)

    def _schedule_task_on_time(
        self,
        task: WorkflowTask,
        workflow: WorkflowDAG,
        placements: Dict[str, str]
    ) -> str:
        """
        Schedules `task` just-in-time via `self.scheduler.schedule_task` if not already pinned.
        """
        if task.name in placements:
            return placements[task.name]

        node_name = self.scheduler.schedule_task(task, workflow, placements, self.env)
        placements[task.name] = node_name
        return node_name

    def _transfer_incoming_data(
        self,
        task: WorkflowTask,
        target_node: str,
        workflow: WorkflowDAG,
        placements: Dict[str, str],
        run_transfer_records: List[DataTransferRecord],
        shared_uploads: Optional[Dict[Tuple[str, str, str, int], simpy.Event]] = None
    ) -> Generator[simpy.Event, None, None]:
        """
        Transfers all incoming data dependencies from each parent's actual execution node
        to `target_node` in parallel.
        """
        incoming_deps = workflow.get_incoming_dependencies(task.name)
        if not incoming_deps:
            return

        transfer_processes = [
            self.env.process(
                self._transfer_single_edge(
                    dep=dep,
                    workflow=workflow,
                    source_node=placements[dep.source_task],
                    target_node=target_node,
                    run_transfer_records=run_transfer_records,
                    shared_uploads=shared_uploads
                )
            )
            for dep in incoming_deps
        ]
        yield simpy.AllOf(self.env, transfer_processes)

    def _transfer_single_edge(
        self,
        dep: DataDependency,
        workflow: WorkflowDAG,
        source_node: str,
        target_node: str,
        run_transfer_records: List[DataTransferRecord],
        shared_uploads: Optional[Dict[Tuple[str, str, str, int], simpy.Event]] = None
    ) -> Generator[simpy.Event, None, None]:
        """
        Performs a single edge data transfer through the DataTransferController and records
        telemetry for the current workflow run.
        """
        mode = workflow.get_effective_mode(dep)
        record = yield from self.controller.transfer(
            env=self.env,
            mode=mode,
            source_node=source_node,
            target_node=target_node,
            size_bytes=dep.size_bytes,
            source_task=dep.source_task,
            target_task=dep.target_task,
            shared_uploads=shared_uploads
        )
        run_transfer_records.append(record)

    def _prepare_execution_target(
        self,
        task: WorkflowTask,
        planned_node: str,
        task_executors: Optional[Dict[str, Callable]]
    ) -> Generator[simpy.Event, None, Tuple[Optional[SimFunctionReplica], str, float]]:
        """
        Resolves a RUNNING replica on `planned_node` (or deploys one via Skippy) before incoming
        data transfer begins, returning `(target_replica, actual_node, cold_start_duration)`.
        """
        if task_executors and task.name in task_executors:
            return None, planned_node, 0.0

        if getattr(self.env, "faas", None) is not None:
            faas = self.env.faas
            if hasattr(faas, "deployment_service") and faas.deployment_service.exists(task.function_name):
                t_prep_start = self.env.now
                target_replica = yield from self._resolve_or_deploy_replica(faas, task.function_name, planned_node)
                cold_start_duration = self.env.now - t_prep_start
                actual_node = (
                    target_replica.node.name
                    if target_replica is not None and target_replica.node is not None
                    else planned_node
                )
                return target_replica, actual_node, cold_start_duration

        return None, planned_node, 0.0

    def _execute_computation(
        self,
        task: WorkflowTask,
        actual_node: str,
        target_replica: Optional[SimFunctionReplica],
        task_executors: Optional[Dict[str, Callable]]
    ) -> Generator[simpy.Event, None, float]:
        """
        Executes pure function computation on `actual_node` (after replica readiness and data arrival)
        and returns pure compute `duration`.
        """
        t_start = self.env.now

        # 1. Custom executor hook
        if task_executors and task.name in task_executors:
            res = yield from task_executors[task.name](task, actual_node)
            return (self.env.now - t_start) if res is None else float(res)

        # 2. Native faas-sim invocation pipeline (targets the resolved replica)
        if target_replica is not None and getattr(self.env, "faas", None) is not None:
            req_kwargs = {k: v for k, v in task.kwargs.items() if k in ("size", "client")}
            request = FunctionRequest(
                name=task.function_name,
                start=self.env.now,
                replica=target_replica,
                **req_kwargs
            )
            yield from self.env.faas.invoke(request)
            return self.env.now - t_start

        # 3. Default simulated delay
        yield self.env.timeout(self.default_task_duration)
        return self.env.now - t_start

    def _resolve_or_deploy_replica(
        self,
        faas,
        function_name: str,
        node_name: str,
        poll_interval: float = 0.005
    ) -> Generator[simpy.Event, None, Optional[SimFunctionReplica]]:
        """
        Finds a RUNNING replica of `function_name` on `node_name`.
        If none is running on `node_name` and no deployment for `(function_name, node_name)`
        is currently in progress, instructs `WorkflowSkippyScheduler` (`env.scheduler`)
        to place a new replica on `node_name`, triggers `faas.scale_up(function_name, 1)`,
        and waits until the replica transitions to `RUNNING`.
        """
        running_on_node = [
            r for r in faas.get_replicas(function_name, state=FunctionReplicaState.RUNNING)
            if r.node is not None and r.node.name == node_name
        ]
        if running_on_node:
            return running_on_node[0]

        starting_on_node = [
            r for r in faas.get_replicas(function_name, running=False)
            if r.node is not None and r.node.name == node_name and r.state == FunctionReplicaState.PENDING
        ]

        key = (function_name, node_name)
        self._pending_waiters[key] += 1

        try:
            if self._pending_waiters[key] == 1:
                if starting_on_node:
                    self._pending_replicas[key] = starting_on_node[0]
                else:
                    before_ids = {
                        r.replica_id
                        for r in faas.replica_service.get_function_replicas_of_deployment(function_name, running=False)
                    }
                    if isinstance(getattr(self.env, "scheduler", None), WorkflowSkippyScheduler):
                        self.env.scheduler.enqueue_placement(function_name, node_name)
                    yield from faas.scale_up(function_name, 1)
                    all_reps = faas.replica_service.get_function_replicas_of_deployment(function_name, running=False)
                    new_reps = [
                        r for r in all_reps
                        if r.replica_id not in before_ids and r.replica_id not in self._claimed_replica_ids
                    ]
                    if new_reps:
                        claimed = new_reps[0]
                        self._claimed_replica_ids.add(claimed.replica_id)
                        self._pending_replicas[key] = claimed
                    else:
                        self._pending_replicas[key] = None

            while True:
                running_replicas = faas.get_replicas(function_name, state=FunctionReplicaState.RUNNING)
                for r in running_replicas:
                    if r.node is not None and r.node.name == node_name:
                        return r

                tracked = self._pending_replicas.get(key)
                if tracked is not None and tracked.state == FunctionReplicaState.RUNNING:
                    return tracked

                pending = [
                    r for r in faas.replica_service.get_function_replicas_of_deployment(function_name, running=False)
                    if r.state in (FunctionReplicaState.CONCEIVED, FunctionReplicaState.PENDING)
                ]
                if not pending and key in self._pending_replicas:
                    if running_replicas:
                        return running_replicas[-1]
                    raise RuntimeError(
                        f"Failed to schedule or start any replica for function '{function_name}' "
                        f"(requested node '{node_name}')."
                    )
                yield self.env.timeout(poll_interval)
        finally:
            self._pending_waiters[key] -= 1
            if self._pending_waiters[key] <= 0:
                self._pending_waiters.pop(key, None)
                self._pending_replicas.pop(key, None)

    def _initialize_placements(
        self,
        task_placements: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Validates and returns any explicit caller-specified task placements.
        Unpinned tasks are left empty and scheduled on-time as the workflow runs.
        """
        placements: Dict[str, str] = {}
        if task_placements:
            for task_name, target_node in task_placements.items():
                if self.env.topology and self.env.topology.find_node(target_node) is None:
                    raise ValueError(f"Placement node '{target_node}' for task '{task_name}' not found in topology.")
                placements[task_name] = target_node
        return placements

    def _compile_results(
        self,
        workflow: WorkflowDAG,
        t_start: float,
        task_records: Dict[str, WorkflowTaskExecutionRecord],
        run_transfer_records: List[DataTransferRecord]
    ) -> WorkflowExecutionResult:
        """
        Assembles the final metrics report for this workflow run.
        """
        t_end = self.env.now
        makespan = t_end - t_start
        sorted_transfers = sorted(run_transfer_records, key=lambda r: r.start_time)

        total_compute = sum(r.duration for r in task_records.values())
        total_cold_start = sum(r.cold_start_duration for r in task_records.values())
        total_transfer_time = sum(r.duration for r in sorted_transfers)
        total_bytes = sum(r.size_bytes for r in sorted_transfers)

        return WorkflowExecutionResult(
            workflow_name=workflow.name,
            start_time=t_start,
            end_time=t_end,
            makespan=makespan,
            total_compute_time=total_compute,
            total_data_transfer_time=total_transfer_time,
            total_data_bytes_transferred=total_bytes,
            task_records=task_records,
            transfer_records=sorted_transfers,
            total_cold_start_time=total_cold_start
        )
