"""
Workflow and Data Passing module for faas-sim.

Provides abstractions for modeling serverless workflows (DAGs) and simulating
different data passing mechanisms (Direct Peer-to-Peer, Remote Storage, Colocated).
"""

from .model import WorkflowDAG, WorkflowTask, DataDependency
from .channel import DataPassingMode, DataChannel, DirectDataChannel, RemoteStorageDataChannel, ColocatedDataChannel, DataTransferRecord

__all__ = [
    'WorkflowDAG',
    'WorkflowTask',
    'DataDependency',
    'DataPassingMode',
    'DataChannel',
    'DirectDataChannel',
    'RemoteStorageDataChannel',
    'ColocatedDataChannel',
    'DataTransferRecord',
]
