import collections
import logging
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Set, Union, Callable

import simpy
from ether.core import Node
from faas.system.core import FunctionRequest

from sim.core import Environment
from sim.workflow.model import WorkflowDAG, WorkflowTask, DataDependency, DataPassingMode
from sim.workflow.channel import DataTransferController, DataTransferRecord

logger = logging.getLogger(__name__)


@dataclass
class WorkflowTaskExecutionRecord:
    """
    Telemetry record of a single task execution in a workflow.
    """
    task_name: str
    function_name: str
    node_name: str
    start_time: float
    end_time: float
    duration: float
    status: str = "SUCCESS"


@dataclass
class WorkflowExecutionResult:
    """
    Summary metrics of an executed workflow.

    - makespan: Total wall-clock time elapsed from workflow start to finish.
                Includes compute time, data transfer time, and concurrency overlap.
    - total_compute_time: Arithmetic sum of pure function execution durations
                          across all tasks (excludes network transfer time).
    - total_data_transfer_time: Arithmetic sum of all intermediate data transfer durations.
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

    Coordinates the execution of serverless workflow DAGs by directly integrating
    with faas-sim's native `FaasSystem` invocation pipeline and gating task execution
    on intermediate data arrivals via `DataTransferController`.
    """

    def __init__(
        self,
        env: Environment,
        controller: Optional[DataTransferController] = None,
        storage_node: Optional[Union[str, Node]] = None,
        default_task_duration: float = 0.5
    ):
        """
        :param env: The faas-sim Environment.
        :param controller: The DataTransferController routing data flows.
        :param storage_node: Central storage node reference for remote storage transfers.
        :param default_task_duration: Fallback duration if running standalone without FaasSystem.
        """
        self.env: Environment = env
        self.controller: DataTransferController = controller or DataTransferController(storage_node=storage_node)
        self.default_task_duration: float = default_task_duration

    def execute_workflow(
        self,
        workflow: WorkflowDAG,
        task_placements: Optional[Dict[str, str]] = None,
        task_executors: Optional[Dict[str, Callable[[WorkflowTask, str], Generator[simpy.Event, None, float]]]] = None
    ) -> Generator[simpy.Event, None, WorkflowExecutionResult]:
        """
        Main entry point to execute a workflow DAG in the simulation.

        1. Resolves task-to-node placements.
        2. Sets up dependency synchronization events.
        3. Spawns tasks as concurrent SimPy processes (tasks wait for their input data).
        4. Awaits completion of all terminal tasks and returns execution metrics.
        """
        t_start = self.env.now
        logger.info("[WorkflowEngine] Starting workflow '%s' at t=%.4f", workflow.name, t_start)

        placements = self._resolve_placements(workflow, task_placements)
        data_ready_events = self._initialize_data_events(workflow)
        task_records: Dict[str, WorkflowTaskExecutionRecord] = {}

        # Run all tasks as concurrent SimPy processes
        task_processes = [
            self.env.process(
                self._run_task(
                    task=task,
                    node_name=placements[task.name],
                    workflow=workflow,
                    placements=placements,
                    data_ready_events=data_ready_events,
                    task_records=task_records,
                    task_executors=task_executors
                )
            )
            for task in workflow.tasks.values()
        ]

        # Await completion of all tasks in the workflow
        yield simpy.AllOf(self.env, task_processes)

        return self._compile_results(workflow, t_start, task_records)

    def _run_task(
        self,
        task: WorkflowTask,
        node_name: str,
        workflow: WorkflowDAG,
        placements: Dict[str, str],
        data_ready_events: Dict[str, Dict[str, simpy.Event]],
        task_records: Dict[str, WorkflowTaskExecutionRecord],
        task_executors: Optional[Dict[str, Callable]]
    ) -> Generator[simpy.Event, None, None]:
        """
        Lifecycle of a single task in the workflow:
        Step 1: Wait for all incoming data dependencies to arrive.
        Step 2: Invoke the function (via faas-sim FaasSystem or simulated executor).
        Step 3: Dispatch outgoing data to all downstream child tasks.
        """
        # Step 1: Wait for input data
        yield from self._wait_for_incoming_data(task.name, data_ready_events)

        # Step 2: Invoke computation
        t_task_start = self.env.now
        duration = yield from self._invoke_function(task, node_name, task_executors)
        t_task_end = self.env.now

        task_records[task.name] = WorkflowTaskExecutionRecord(
            task_name=task.name,
            function_name=task.function_name,
            node_name=node_name,
            start_time=t_task_start,
            end_time=t_task_end,
            duration=duration
        )

        # Step 3: Transmit output data to downstream tasks
        yield from self._dispatch_outgoing_data(task.name, node_name, workflow, placements, data_ready_events)

    def _wait_for_incoming_data(
        self,
        task_name: str,
        data_ready_events: Dict[str, Dict[str, simpy.Event]]
    ) -> Generator[simpy.Event, None, None]:
        """
        Awaits all incoming data events required before this task can start.
        """
        incoming = data_ready_events.get(task_name, {})
        if incoming:
            logger.debug(
                "[WorkflowEngine] Task '%s' waiting for data from: %s at t=%.4f",
                task_name, list(incoming.keys()), self.env.now
            )
            yield simpy.AllOf(self.env, list(incoming.values()))

    def _invoke_function(
        self,
        task: WorkflowTask,
        node_name: str,
        task_executors: Optional[Dict[str, Callable]]
    ) -> Generator[simpy.Event, None, float]:
        """
        Executes the function computation.
        Prioritizes:
          1. Custom user executor hook if provided.
          2. faas-sim's native `env.faas.invoke(FunctionRequest)` if deployed.
          3. Default simulated task duration.
        """
        t_start = self.env.now

        # 1. Custom executor hook
        if task_executors and task.name in task_executors:
            res = yield from task_executors[task.name](task, node_name)
            return (self.env.now - t_start) if res is None else float(res)

        # 2. Native faas-sim invocation pipeline
        if getattr(self.env, "faas", None) is not None:
            faas = self.env.faas
            if hasattr(faas, "deployment_service") and faas.deployment_service.exists(task.function_name):
                # Reuse faas-sim's native FunctionRequest object
                request = FunctionRequest(name=task.function_name, start=self.env.now, **task.kwargs)
                yield from faas.invoke(request)
                return self.env.now - t_start

        # 3. Default simulated delay
        yield self.env.timeout(self.default_task_duration)
        return self.env.now - t_start

    def _dispatch_outgoing_data(
        self,
        source_task: str,
        source_node: str,
        workflow: WorkflowDAG,
        placements: Dict[str, str],
        data_ready_events: Dict[str, Dict[str, simpy.Event]]
    ) -> Generator[simpy.Event, None, None]:
        """
        Asynchronously triggers data transfers to all downstream dependent tasks.
        """
        downstream = workflow.get_downstream_dependencies(source_task)
        for dep in downstream:
            target_node = placements[dep.target_task]
            ready_event = data_ready_events[dep.target_task][source_task]

            # Spawn transfer process concurrently so multiple children receive data in parallel
            self.env.process(
                self._transfer_data(
                    dep=dep,
                    source_node=source_node,
                    target_node=target_node,
                    ready_event=ready_event
                )
            )

        # Allow immediate continuation
        yield self.env.timeout(0)

    def _transfer_data(
        self,
        dep: DataDependency,
        source_node: str,
        target_node: str,
        ready_event: simpy.Event
    ) -> Generator[simpy.Event, None, None]:
        """
        Performs the data transfer through the DataTransferController and signals readiness.
        """
        mode = dep.mode or DataPassingMode.DIRECT
        yield from self.controller.transfer(
            env=self.env,
            mode=mode,
            source_node=source_node,
            target_node=target_node,
            size_bytes=dep.size_bytes,
            source_task=dep.source_task,
            target_task=dep.target_task
        )
        ready_event.succeed()

    def _initialize_data_events(self, workflow: WorkflowDAG) -> Dict[str, Dict[str, simpy.Event]]:
        """
        Creates a SimPy event for every data dependency: incoming_events[target_task][source_task].
        """
        events: Dict[str, Dict[str, simpy.Event]] = collections.defaultdict(dict)
        for dep in workflow.dependencies:
            events[dep.target_task][dep.source_task] = self.env.event()
        return events

    def _resolve_placements(
        self,
        workflow: WorkflowDAG,
        task_placements: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Resolves node assignments for each task in the workflow.
        """
        placements: Dict[str, str] = {}
        available_nodes = [n.name for n in self.env.topology.get_nodes()] if self.env.topology else ["default_node"]

        for idx, task_name in enumerate(workflow.tasks):
            if task_placements and task_name in task_placements:
                target_node = task_placements[task_name]
                if self.env.topology and self.env.topology.find_node(target_node) is None:
                    raise ValueError(f"Placement node '{target_node}' for task '{task_name}' not found in topology.")
                placements[task_name] = target_node
            else:
                placements[task_name] = available_nodes[idx % len(available_nodes)]

        return placements

    def _compile_results(
        self,
        workflow: WorkflowDAG,
        t_start: float,
        task_records: Dict[str, WorkflowTaskExecutionRecord]
    ) -> WorkflowExecutionResult:
        """
        Assembles the final metrics report for the workflow run.
        """
        t_end = self.env.now
        makespan = t_end - t_start
        all_transfers = self.controller.get_all_records()

        total_compute = sum(r.duration for r in task_records.values())
        total_transfer_time = sum(r.duration for r in all_transfers)
        total_bytes = sum(r.size_bytes for r in all_transfers)

        return WorkflowExecutionResult(
            workflow_name=workflow.name,
            start_time=t_start,
            end_time=t_end,
            makespan=makespan,
            total_compute_time=total_compute,
            total_data_transfer_time=total_transfer_time,
            total_data_bytes_transferred=total_bytes,
            task_records=task_records,
            transfer_records=all_transfers
        )
