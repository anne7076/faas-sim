import collections
import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any


class DataPassingMode(str, enum.Enum):
    """
    Data passing strategies for intermediate serverless workflow communication.
    """
    DIRECT = "direct"                  # Direct peer-to-peer transmission (Node A -> Node B)
    REMOTE_STORAGE = "remote_storage"  # Triangular transmission (Node A -> Storage -> Node B)
    COLOCATED = "colocated"            # In-memory/IPC transfer on the same node


@dataclass
class WorkflowTask:
    """
    Represents an individual step or task in a serverless workflow.
    """
    name: str
    function_name: str
    image: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, WorkflowTask):
            return self.name == other.name
        return False


@dataclass
class DataDependency:
    """
    Represents a directed data dependency edge (source_task -> target_task)
    carrying intermediate data of size `size_bytes`.
    Optionally specifies a dedicated `mode`; if None, inherits the workflow's default mode.
    """
    source_task: str
    target_task: str
    size_bytes: int
    mode: Optional[DataPassingMode] = None

    def __post_init__(self):
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative, got {self.size_bytes}")


class WorkflowDAG:
    """
    Directed Acyclic Graph (DAG) representing a serverless workflow.
    Nodes are WorkflowTasks and edges are DataDependencies.
    """

    def __init__(self, name: str = "workflow", default_mode: DataPassingMode = DataPassingMode.DIRECT):
        self.name: str = name
        self.default_mode: DataPassingMode = default_mode
        self.tasks: Dict[str, WorkflowTask] = {}
        self.dependencies: List[DataDependency] = []
        self._downstream: Dict[str, List[DataDependency]] = collections.defaultdict(list)
        self._upstream: Dict[str, List[DataDependency]] = collections.defaultdict(list)

    def add_task(self, task: WorkflowTask) -> 'WorkflowDAG':
        """Adds a task to the workflow."""
        if task.name in self.tasks:
            raise ValueError(f"Task with name '{task.name}' already exists in workflow '{self.name}'")
        self.tasks[task.name] = task
        return self

    def add_dependency(
        self,
        source_task: str,
        target_task: str,
        size_bytes: int = 0,
        mode: Optional[DataPassingMode] = None
    ) -> 'WorkflowDAG':
        """
        Adds a directed data dependency: source_task -> target_task with `size_bytes`.
        Optionally override the data passing mode for this specific edge.
        """
        if source_task not in self.tasks:
            raise ValueError(f"Source task '{source_task}' does not exist in workflow '{self.name}'")
        if target_task not in self.tasks:
            raise ValueError(f"Target task '{target_task}' does not exist in workflow '{self.name}'")
        if source_task == target_task:
            raise ValueError(f"Self-dependencies are not allowed (task '{source_task}')")

        dep = DataDependency(
            source_task=source_task,
            target_task=target_task,
            size_bytes=size_bytes,
            mode=mode or self.default_mode
        )
        self.dependencies.append(dep)
        self._downstream[source_task].append(dep)
        self._upstream[target_task].append(dep)

        # Validate that no cycles are introduced
        self.get_execution_order()
        return self

    def get_downstream_dependencies(self, task_name: str) -> List[DataDependency]:
        """Returns outgoing data dependencies from task_name."""
        return list(self._downstream.get(task_name, []))

    def get_incoming_dependencies(self, task_name: str) -> List[DataDependency]:
        """Returns incoming data dependencies required by task_name."""
        return list(self._upstream.get(task_name, []))

    def get_root_tasks(self) -> List[WorkflowTask]:
        """Returns all tasks with no incoming dependencies (entry points)."""
        return [task for name, task in self.tasks.items() if not self._upstream.get(name)]

    def get_terminal_tasks(self) -> List[WorkflowTask]:
        """Returns all tasks with no outgoing dependencies (exit points)."""
        return [task for name, task in self.tasks.items() if not self._downstream.get(name)]

    def get_execution_order(self) -> List[str]:
        """
        Computes a topological sort of task names using Kahn's algorithm.
        Raises ValueError if the workflow contains a cycle.
        """
        in_degree: Dict[str, int] = {name: 0 for name in self.tasks}
        for dep in self.dependencies:
            in_degree[dep.target_task] += 1

        queue = collections.deque([name for name, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for dep in self._downstream.get(node, []):
                in_degree[dep.target_task] -= 1
                if in_degree[dep.target_task] == 0:
                    queue.append(dep.target_task)

        if len(order) != len(self.tasks):
            raise ValueError(f"Workflow '{self.name}' contains a cycle and is not a valid DAG.")

        return order

    def __len__(self) -> int:
        return len(self.tasks)
