import pytest
import simpy

import examples.basic.main as basic
from sim.core import Environment
from sim.factory.flow import UninterruptingFlowFactory
from sim.workflow.model import WorkflowDAG, WorkflowTask, DataDependency, DataPassingMode
from sim.workflow.channel import (
    DirectDataChannel,
    RemoteStorageDataChannel,
    ColocatedDataChannel,
    DataTransferController,
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

    def test_per_edge_data_passing_mode(self):
        dag = WorkflowDAG("hybrid-dag", default_mode=DataPassingMode.DIRECT)
        t1 = WorkflowTask(name="A", function_name="fn-a")
        t2 = WorkflowTask(name="B", function_name="fn-b")
        t3 = WorkflowTask(name="C", function_name="fn-c")

        dag.add_task(t1).add_task(t2).add_task(t3)
        dag.add_dependency("A", "B", size_bytes=1024, mode=DataPassingMode.COLOCATED)
        dag.add_dependency("B", "C", size_bytes=2048)  # Should inherit default DIRECT

        deps_from_a = dag.get_downstream_dependencies("A")
        assert deps_from_a[0].mode == DataPassingMode.COLOCATED

        deps_from_b = dag.get_downstream_dependencies("B")
        assert deps_from_b[0].mode == DataPassingMode.DIRECT


class TestDataChannels:

    def test_channel_comparison(self):
        env = Environment()
        env.topology = basic.example_topology()
        env.flow_factory = UninterruptingFlowFactory()

        # Dynamically retrieve edge nodes and cloud server node from topology
        nodes = list(env.topology.get_nodes())
        edge_nodes = [n for n in nodes if 'rpi' in n.name]
        cloud_nodes = [n for n in nodes if 'server' in n.name]

        node_a = edge_nodes[0].name
        node_b = edge_nodes[1].name
        cloud_storage = cloud_nodes[0].name

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

            # 2. Direct P2P transfer: node_a -> node_b
            direct_ch = DirectDataChannel()
            rec_direct = yield from direct_ch.transfer(
                env, source_node=node_a, target_node=node_b, size_bytes=payload_size,
                source_task="taskA", target_task="taskB"
            )
            results["direct"] = rec_direct.duration

            # 3. Remote storage transfer (Triangular): node_a -> cloud_storage -> node_b
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

    def test_data_transfer_controller_hybrid(self):
        env = Environment()
        env.topology = basic.example_topology()
        env.flow_factory = UninterruptingFlowFactory()

        # Dynamically retrieve edge nodes and cloud server node
        nodes = list(env.topology.get_nodes())
        edge_nodes = [n for n in nodes if 'rpi' in n.name]
        cloud_nodes = [n for n in nodes if 'server' in n.name]

        node_a = edge_nodes[0].name
        node_b = edge_nodes[1].name
        storage = cloud_nodes[0].name

        controller = DataTransferController(storage_node=storage)
        results = []

        def hybrid_process():
            # Hop 1: Direct P2P
            rec1 = yield from controller.transfer(
                env=env,
                mode=DataPassingMode.DIRECT,
                source_node=node_a,
                target_node=node_b,
                size_bytes=1024 * 1024,
                source_task="task1",
                target_task="task2"
            )
            results.append(rec1)

            # Hop 2: Colocated
            rec2 = yield from controller.transfer(
                env=env,
                mode=DataPassingMode.COLOCATED,
                source_node=node_b,
                target_node=node_b,
                size_bytes=1024 * 1024,
                source_task="task2",
                target_task="task3"
            )
            results.append(rec2)

            # Hop 3: Remote storage
            rec3 = yield from controller.transfer(
                env=env,
                mode=DataPassingMode.REMOTE_STORAGE,
                source_node=node_b,
                target_node=storage,
                size_bytes=1024 * 1024,
                source_task="task3",
                target_task="task4"
            )
            results.append(rec3)

        env.process(hybrid_process())
        env.run()

        assert len(results) == 3
        all_records = controller.get_all_records()
        assert len(all_records) == 3
        assert [r.mode for r in all_records] == [
            DataPassingMode.DIRECT,
            DataPassingMode.COLOCATED,
            DataPassingMode.REMOTE_STORAGE
        ]
