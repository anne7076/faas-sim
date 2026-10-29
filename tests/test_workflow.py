import pytest
import simpy

import examples.basic.main as basic
from sim.core import Environment
from sim.factory.flow import UninterruptingFlowFactory
from sim.workflow.model import WorkflowDAG, WorkflowTask, DataDependency
from sim.workflow.channel import (
    DataPassingMode,
    DirectDataChannel,
    RemoteStorageDataChannel,
    ColocatedDataChannel,
)


class TestWorkflowDAG:

    def test_dag_construction_and_ordering(self):
        dag = WorkflowDAG("test-ml-pipeline")
        t1 = WorkflowTask(name="preprocess", function_name="data-prep")
        t2 = WorkflowTask(name="train", function_name="model-training")
        t3 = WorkflowTask(name="evaluate", function_name="model-eval")

        dag.add_task(t1).add_task(t2).add_task(t3)
        dag.add_dependency("preprocess", "train", size_bytes=10 * 1024 * 1024)
        dag.add_dependency("train", "evaluate", size_bytes=2 * 1024 * 1024)

        order = dag.get_execution_order()
        assert order == ["preprocess", "train", "evaluate"]
        assert dag.get_root_tasks() == [t1]
        assert dag.get_terminal_tasks() == [t3]

    def test_cycle_detection(self):
        dag = WorkflowDAG("cycle-test")
        dag.add_task(WorkflowTask(name="A", function_name="fn-a"))
        dag.add_task(WorkflowTask(name="B", function_name="fn-b"))
        dag.add_task(WorkflowTask(name="C", function_name="fn-c"))

        dag.add_dependency("A", "B", size_bytes=100)
        dag.add_dependency("B", "C", size_bytes=100)

        with pytest.raises(ValueError, match="contains a cycle"):
            dag.add_dependency("C", "A", size_bytes=100)

    def test_self_dependency_rejected(self):
        dag = WorkflowDAG("self-dep-test")
        dag.add_task(WorkflowTask(name="A", function_name="fn-a"))
        with pytest.raises(ValueError, match="Self-dependencies are not allowed"):
            dag.add_dependency("A", "A", size_bytes=50)


class TestDataChannels:

    def test_channel_comparison(self):
        env = Environment()
        env.topology = basic.example_topology()
        env.flow_factory = UninterruptingFlowFactory()

        # Nodes from UrbanSensing topology:
        # rpi3_0 (edge), rpi3_1 (edge), server_0 (cloud storage)
        node_a = "rpi3_0"
        node_b = "rpi3_1"
        cloud_storage = "server_0"

        payload_size = 5 * 1024 * 1024  # 5 MB
        results = {}

        def simulation_process():
            # 1. Colocated transfer on node_a
            colocated_ch = ColocatedDataChannel()
            rec_coloc = yield from colocated_ch.transfer(
                env, source_node=node_a, target_node=node_a, size_bytes=payload_size,
                source_task="taskA", target_task="taskB"
            )
            results["colocated"] = rec_coloc.duration

            # 2. Direct P2P transfer: rpi3_0 -> rpi3_1
            direct_ch = DirectDataChannel()
            rec_direct = yield from direct_ch.transfer(
                env, source_node=node_a, target_node=node_b, size_bytes=payload_size,
                source_task="taskA", target_task="taskB"
            )
            results["direct"] = rec_direct.duration

            # 3. Remote storage transfer (Triangular): rpi3_0 -> server_0 -> rpi3_1
            remote_ch = RemoteStorageDataChannel(storage_node=cloud_storage)
            rec_remote = yield from remote_ch.transfer(
                env, source_node=node_a, target_node=node_b, size_bytes=payload_size,
                source_task="taskA", target_task="taskB"
            )
            results["remote"] = rec_remote.duration

        env.process(simulation_process())
        env.run()

        # Colocated should be virtually instantaneous (in-memory)
        assert results["colocated"] < results["direct"]
        # Direct passing should be significantly faster than triangular transfer to cloud storage
        assert results["direct"] < results["remote"]
        # Remote storage takes 2 hops (edge -> cloud, cloud -> edge)
        assert results["remote"] > results["direct"] * 1.5
