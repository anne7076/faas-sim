"""
Workflow and Data Passing module for faas-sim.

Provides abstractions for modeling serverless workflows (DAGs) and simulating
different data passing mechanisms (Direct Peer-to-Peer, Remote Storage, Colocated),
with support for heterogeneous/hybrid per-edge passing strategies and automated
SimPy workflow orchestration.
"""

from .model import WorkflowDAG, WorkflowTask, DataDependency, DataPassingMode
from .channel import (
    DataChannel,
    DirectDataChannel,
    RemoteStorageDataChannel,
    ColocatedDataChannel,
    DataTransferController,
    DataTransferRecord,
)
from .engine import (
    WorkflowEngine,
    WorkflowExecutionResult,
    WorkflowTaskExecutionRecord,
)
from .scheduler import (
    WorkflowScheduler,
    RoundRobinWorkflowScheduler,
    DataAwareWorkflowScheduler,
)

__all__ = [
    'WorkflowDAG',
    'WorkflowTask',
    'DataDependency',
    'DataPassingMode',
    'DataChannel',
    'DirectDataChannel',
    'RemoteStorageDataChannel',
    'ColocatedDataChannel',
    'DataTransferController',
    'DataTransferRecord',
    'WorkflowEngine',
    'WorkflowExecutionResult',
    'WorkflowTaskExecutionRecord',
    'WorkflowScheduler',
    'RoundRobinWorkflowScheduler',
    'DataAwareWorkflowScheduler',
]
